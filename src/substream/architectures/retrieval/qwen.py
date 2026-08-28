"""Qwen3-VL-Embedding-8B backend (Qwen/Qwen3-VL-Embedding-8B). Requires an environment
with transformers >= 4.57. Uses the vendored official wrapper (qwen3_vl_embedding.Qwen3VLEmbedder):
last non-pad token pooling, RIGHT padding, L2-normalized output; total_pixels lowered to a 16-frame budget."""
from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"
TOTAL_PIXELS = 3 * 768 * 32 * 32     # ~1/3 of default: 16 frames, not 64
QUERY_INSTR = "Retrieve the video clip that matches the query."


def _query_text(text: str, event: str | None) -> str:
    target = (event or "").replace("_", " ").strip() or text
    return f"the action: {target}"


class Backend:
    def __init__(self):
        from substream.architectures.retrieval.qwen3_vl_embedding import Qwen3VLEmbedder
        self.model = Qwen3VLEmbedder(
            model_name_or_path=MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            total_pixels=TOTAL_PIXELS,
        )

    def embed_video_files(self, paths, batch=4):
        chunks = []
        for i in range(0, len(paths), batch):
            inputs = [{"video": p, "fps": 1, "max_frames": 16}
                      for p in paths[i:i + batch]]
            emb = self.model.process(inputs)
            chunks.append(np.asarray(emb, dtype=np.float32)
                          if not torch.is_tensor(emb)
                          else emb.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def embed_text(self, text, event=None):
        emb = self.model.process([{"text": _query_text(text, event),
                                   "instruction": QUERY_INSTR}])
        arr = (emb.float().cpu().numpy() if torch.is_tensor(emb)
               else np.asarray(emb, dtype=np.float32))
        return arr[0]
