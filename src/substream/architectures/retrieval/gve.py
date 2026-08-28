"""GVE-7B backend (Alibaba-NLP/GVE-7B). Requires an environment with transformers
4.50.x: the model's remote code imports QWEN2_5_VL_INPUTS_DOCSTRING, which
transformers 5 removed. Last-token pooling with LEFT padding, then L2-normalize."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

MODEL_ID = "Alibaba-NLP/GVE-7B"
MAX_PIXELS = 200 * 28 * 28          # card's per-frame budget


def _query_text(text: str, event: str | None) -> str:
    # "open_fridge" -> "open fridge"; falls back to the raw question.
    target = (event or "").replace("_", " ").strip() or text
    return f"Find the video clip showing this action: {target}."


class Backend:
    def __init__(self):
        from transformers import AutoModel, AutoProcessor
        # flash_attention_2 is REQUIRED, not an optimisation: without it the
        # Qwen2.5-VL vision tower falls back to SDPA with a materialised mask
        # over the concatenated patch sequence of the whole batch -- observed
        # 2,025 GiB allocation at batch=32 (32 clips x 16 frames x ~200 patches).
        self.model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, device_map="auto",
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_fast=True)
        self.processor.tokenizer.padding_side = "left"   # required for [:, -1, :]

    @torch.no_grad()
    def _embed_messages(self, messages_batch):
        from qwen_vl_utils import process_vision_info
        texts = [self.processor.apply_chat_template(
                     m, tokenize=False, add_generation_prompt=True)
                 for m in messages_batch]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages_batch, return_video_kwargs=True)
        # NO truncation: clips at native resolution run ~1,440 tokens, and
        # truncating chops video placeholder tokens off the text while the
        # vision features remain -> "Video features and video tokens do not
        # match" (observed: tokens 9480 vs features 11520 at batch 8).
        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **video_kwargs).to("cuda")
        out = self.model(**inputs)
        emb = F.normalize(out["last_hidden_state"][:, -1, :], p=2, dim=1)
        return emb.float().cpu().numpy()

    def embed_video_files(self, paths, batch=8):
        chunks = []
        for i in range(0, len(paths), batch):
            msgs = [[{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": [
                         {"type": "video", "video": p,
                          "max_pixels": MAX_PIXELS, "fps": 1.0, "max_frames": 16},
                         {"type": "text", "text": "Describe this video."}]}]
                    for p in paths[i:i + batch]]
            chunks.append(self._embed_messages(msgs))
        return np.concatenate(chunks, axis=0)

    def embed_text(self, text, event=None):
        msg = [[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": _query_text(text, event)}]}]]
        return self._embed_messages(msg)[0]
