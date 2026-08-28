"""The single table of evaluable models.

Each entry maps a CLI alias to everything the harness needs to know about a
model: where its wrapper class lives, which checkpoint it loads, which frame
sampler and Python environment it needs, and its batching limits. Adding a
model means adding one row here and one wrapper module under
``substream/models/``.

``kind`` distinguishes three things that answer questions:
  * ``vlm`` — a locally loaded vision-language checkpoint,
  * ``api`` — a hosted model reached over HTTP (no GPU),
  * ``architecture`` — a system composed of models plus retrieval or memory
    (see ``substream/architectures/``).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    module: str
    class_name: str
    display_name: str
    kind: str  # "vlm" | "api" | "architecture"
    env: str  # environment name for bin/envctl; "api" models run anywhere
    checkpoint: str | None = None
    sampler: str | None = None  # substream.samplers module for video input
    gpu_mem_aware: bool = False  # wrapper accepts a max_gpu_mem kwarg
    text_batching: bool = True
    video_batching: bool = True
    video_batch_size: int | None = None  # None = the global default
    max_video_bucket: int | None = None  # longest video bucket the model can run
    takes_max_tokens_at_init: bool = False  # constructor wants generation_max_tokens
    init_kwargs: dict = field(default_factory=dict)


def _internvl(checkpoint: str, size: str, thinking: bool) -> ModelSpec:
    return ModelSpec(
        module="substream.models.internvl_3_5",
        class_name="InternVL35ThinkingModel" if thinking else "InternVL35Model",
        display_name=f"InternVL3.5{'-Thinking' if thinking else ''} ({size})",
        kind="vlm",
        env="full-stack",
        checkpoint=checkpoint,
        sampler="internvl_sampler.InternVLFrameSampler",
        gpu_mem_aware=True,
        max_video_bucket=512 if size == "8B" else None,
        takes_max_tokens_at_init=True,
    )


REGISTRY: dict[str, ModelSpec] = {
    "qwen_full": ModelSpec(
        module="substream.models.qwen2_5_vl",
        class_name="QwenFullVideo",
        display_name="Qwen2.5-VL (7B)",
        kind="vlm",
        env="full-stack",
        checkpoint="Qwen/Qwen2.5-VL-7B-Instruct",
        sampler="qwen_sampler.QwenFrameSampler",
        gpu_mem_aware=True,
    ),
    "qwen3_full": ModelSpec(
        module="substream.models.qwen3_dense",
        class_name="Qwen3Dense",
        display_name="Qwen3-VL (8B)",
        kind="vlm",
        env="full-stack",
        sampler="qwen_sampler.QwenFrameSampler",
        gpu_mem_aware=True,
        video_batching=False,
        init_kwargs={"thinking": False},
    ),
    "qwen3_full_thinking": ModelSpec(
        module="substream.models.qwen3_dense",
        class_name="Qwen3Dense",
        display_name="Qwen3-VL-Thinking (8B)",
        kind="vlm",
        env="full-stack",
        sampler="qwen_sampler.QwenFrameSampler",
        gpu_mem_aware=True,
        video_batching=False,
        init_kwargs={"thinking": True},
    ),
    "qwen3_omni": ModelSpec(
        module="substream.models.qwen3_omni",
        class_name="Qwen3Omni",
        display_name="Qwen3-Omni (30B, A3B)",
        kind="vlm",
        env="full-stack",
        checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        sampler="qwen3_omni_sampler.Qwen3oFrameSampler",
        gpu_mem_aware=True,
    ),
    "qwen3_segment_refiner": ModelSpec(
        module="substream.models.qwen3_segment_refiner",
        class_name="Qwen3SegmentRefiner",
        display_name="Qwen3 Segment Refiner (8B)",
        kind="vlm",
        env="full-stack",
        sampler="qwen_sampler.QwenFrameSampler",
        gpu_mem_aware=True,
    ),
    "mimo-vl": ModelSpec(
        module="substream.models.mimo_vl",
        class_name="MimoVLVideo",
        display_name="MiMo-VL (7B)",
        kind="vlm",
        env="full-stack",
        checkpoint="XiaomiMiMo/MiMo-VL-7B-RL",
        sampler="qwen_sampler.QwenFrameSampler",
        gpu_mem_aware=True,
        video_batch_size=4,
    ),
    "glm45v": ModelSpec(
        module="substream.models.glm45v",
        class_name="GLM45V",
        display_name="GLM-4.5V (104B, A12B)",
        kind="vlm",
        env="full-stack",
        checkpoint="zai-org/GLM-4.5V",
        sampler="qwen_sampler.QwenFrameSampler",
        gpu_mem_aware=True,
        video_batch_size=1,
    ),
    "glm45v_fp8": ModelSpec(
        module="substream.models.glm45v_fp8",
        class_name="GLM45VFP8",
        display_name="GLM-4.5V FP8 (104B, A12B)",
        kind="vlm",
        env="full-stack",
        checkpoint="zai-org/GLM-4.5V-FP8",
        sampler="qwen_sampler.QwenFrameSampler",
        gpu_mem_aware=True,
        video_batch_size=1,
    ),
    "internvl-3-5": _internvl("OpenGVLab/InternVL3_5-8B", "8B", thinking=False),
    "internvl-3-5-thinking": _internvl("OpenGVLab/InternVL3_5-8B", "8B", thinking=True),
    "internvl-3-5-30b-a3b": _internvl("OpenGVLab/InternVL3_5-30B-A3B", "30B, A3B", thinking=False),
    "internvl-3-5-30b-a3b-thinking": _internvl(
        "OpenGVLab/InternVL3_5-30B-A3B", "30B, A3B", thinking=True
    ),
    "internvl-3-5-38b": _internvl("OpenGVLab/InternVL3_5-38B", "38B", thinking=False),
    "internvl-3-5-38b-thinking": _internvl("OpenGVLab/InternVL3_5-38B", "38B", thinking=True),
    "minicpm": ModelSpec(
        module="substream.models.minicpm_v_2_6",
        class_name="MiniCPMVideo",
        display_name="MiniCPM-V 2.6 (8B)",
        kind="vlm",
        env="hf450",
        checkpoint="openbmb/MiniCPM-V-2_6",
        sampler="minicpm_v_2_6_sampler.MiniCPMFrameSampler",
        video_batching=False,
    ),
    "minicpm-4-5": ModelSpec(
        module="substream.models.minicpm_v_4_5",
        class_name="MiniCPM45Model",
        display_name="MiniCPM-V 4.5 (9B)",
        kind="vlm",
        env="full-stack",
        checkpoint="openbmb/MiniCPM-V-4_5",
        sampler="minicpm45_sampler.MiniCPM45FrameSampler",
        gpu_mem_aware=True,
    ),
    "longvila": ModelSpec(
        module="substream.models.longvila",
        class_name="LongVILAModel",
        display_name="LongVILA-R1 (7B)",
        kind="vlm",
        env="full-stack",
        checkpoint="Efficient-Large-Model/LongVILA-R1-7B",
        sampler="longvila_sampler.LongVILAFrameSampler",
        gpu_mem_aware=True,
        text_batching=False,
        video_batching=False,
        max_video_bucket=256,
    ),
    "phi_multimodal": ModelSpec(
        module="substream.models.phi_4_mm",
        class_name="PhiMultimodalVideo",
        display_name="Phi-4-MM (6B)",
        kind="vlm",
        env="hf448",
        checkpoint="microsoft/Phi-4-multimodal-instruct",
        sampler="phi_sampler.PhiFrameSampler",
        max_video_bucket=512,
    ),
    "timechat": ModelSpec(
        module="substream.models.timechat",
        class_name="TimeChatOnlineStreaming",
        display_name="TimeChat-Online (7B)",
        kind="vlm",
        env="hf450",
        checkpoint="wyccccc/TimeChatOnline-7B",
        sampler="timechat_sampler.TimeChatFrameSampler",
        gpu_mem_aware=True,
        text_batching=False,
        video_batching=False,
    ),
    "m3_agent": ModelSpec(
        module="substream.architectures.m3_agent",
        class_name="M3Agent",
        display_name="M3-Agent (7B)",
        kind="architecture",
        env="hf451",
        checkpoint="ByteDance-Seed/M3-Agent-Control",
        sampler="m3_agent_sampler.M3AgentFrameSampler",
        text_batching=False,
        video_batching=False,
    ),
    "claude-opus-4-7": ModelSpec(
        module="substream.models.api.claude",
        class_name="ClaudeAPIModel",
        display_name="Claude Opus 4.7",
        kind="api",
        env="api",
        checkpoint="claude-opus-4-7",
        sampler="qwen_sampler.QwenFrameSampler",
        text_batching=False,
        video_batching=False,
    ),
    "claude-opus-4-6": ModelSpec(
        module="substream.models.api.claude",
        class_name="ClaudeAPIModel",
        display_name="Claude Opus 4.6",
        kind="api",
        env="api",
        checkpoint="claude-opus-4-6",
        sampler="qwen_sampler.QwenFrameSampler",
        text_batching=False,
        video_batching=False,
    ),
    "openrouter": ModelSpec(
        module="substream.models.api.openrouter",
        class_name="OpenRouterAPIModel",
        display_name="OpenRouter",
        kind="api",
        env="api",
        sampler="qwen_sampler.QwenFrameSampler",
        text_batching=False,
        video_batching=False,
    ),
    "dummy_eval": ModelSpec(
        module="substream.models.dummy_eval",
        class_name="DummyEvalModel",
        display_name="Dummy (integration tests)",
        kind="vlm",
        env="api",
        sampler="qwen_sampler.QwenFrameSampler",
        text_batching=False,
        video_batching=False,
    ),
}


def load_model_class(alias: str):
    """Import and return the wrapper class for a registry alias."""
    spec = REGISTRY[alias]
    module = importlib.import_module(spec.module)
    return getattr(module, spec.class_name)
