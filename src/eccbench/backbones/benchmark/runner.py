"""Latency benchmarking for MemoryBankDecoder backends."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from eccbench.backbones.data.sequences.common import IGNORE_INDEX


@dataclass
class BenchmarkConfig:
    sequence_lengths: list[int]
    batch_size: int = 1
    repeat: int = 10
    warmup: int = 3
    device: str = "auto"
    query_length: int | None = None


@dataclass
class BenchmarkResult:
    sequence_length: int
    latency_mean_ms: float
    latency_p90_ms: float


class BenchmarkRunner:
    def __init__(
        self,
        model,
        *,
        pad_id: int,
        vocab_size: int,
        task: str,
        config: BenchmarkConfig,
    ) -> None:
        self.model = model.eval()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.task = task
        self.config = config
        if config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.device)
        self.device = device
        self.model.to(device)
        self._long_seq_threshold = 4096
        lengths = sorted(config.sequence_lengths)
        self._ultra_seq_threshold = lengths[-1] if lengths else None
        self._ultra_repeat_runs = 3
        self._max_latency_repeat = 5

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for seq_len in self.config.sequence_lengths:
            results.append(self._measure_sequence(seq_len))
        return results

    def _measure_sequence(self, seq_len: int) -> BenchmarkResult:
        batch = self._build_batch(seq_len)
        self._warmup(batch)
        repeat = self._repeat_for_length(seq_len)
        latency_mean, latency_p90 = self._measure_latency(batch, repeat)
        return BenchmarkResult(
            sequence_length=seq_len,
            latency_mean_ms=latency_mean,
            latency_p90_ms=latency_p90,
        )

    def _repeat_for_length(self, seq_len: int) -> int:
        base = max(1, self.config.repeat)
        if self._ultra_seq_threshold is not None and seq_len >= self._ultra_seq_threshold:
            repeat = min(base, self._ultra_repeat_runs)
        elif seq_len >= self._long_seq_threshold:
            repeat = max(1, math.ceil(base / 2))
        else:
            repeat = base
        return min(repeat, self._max_latency_repeat)

    def _build_batch(
        self,
        seq_len: int,
        *,
        query_len: int | None = None,
    ) -> dict[str, Any]:
        batch_size = self.config.batch_size
        device = self.device
        stream_ids = torch.randint(
            low=1,
            high=max(2, self.vocab_size - 1),
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        query_len = self._resolve_query_length(query_len)
        query_ids = torch.randint(
            low=1,
            high=max(2, self.vocab_size - 1),
            size=(batch_size, query_len),
            device=device,
            dtype=torch.long,
        )
        return self._assemble_sequence_batch(stream_ids, query_ids)

    def _assemble_sequence_batch(
        self,
        stream_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> dict[str, Any]:
        batch_size = stream_ids.size(0)
        stream_len = stream_ids.size(1)
        query_len = query_ids.size(1)
        device = stream_ids.device
        total_len = stream_len + query_len
        if query_len > 0:
            sequence_ids = torch.cat([stream_ids, query_ids], dim=1)
        else:
            sequence_ids = stream_ids.clone()
        padding_mask = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
        lengths = torch.full((batch_size,), total_len, dtype=torch.long, device=device)
        labels = torch.full((batch_size, total_len), IGNORE_INDEX, dtype=torch.long, device=device)
        metadata = [{"stream_length": stream_len} for _ in range(batch_size)]
        sequence = {
            "input_ids": sequence_ids,
            "padding_mask": padding_mask,
            "lengths": lengths,
            "embeddings": None,
            "embedding_mask": None,
        }
        return {
            "sequence": sequence,
            "labels": labels,
            "metadata": metadata,
        }

    def _resolve_query_length(self, override: int | None) -> int:
        if override is not None and override >= 0:
            return int(override)
        query_len = self.config.query_length
        if query_len is None or query_len <= 0:
            query_len = 16
        return int(query_len)

    def _forward_once(self, batch: dict[str, Any]) -> None:
        with torch.inference_mode():
            metadata = batch.get("metadata") or [
                {"stream_length": int(batch["sequence"]["lengths"][0].item())}
            ]
            self.model.compute_sequence_logits(
                batch["sequence"],
                batch["labels"],
                metadata,
            )

    def _warmup(self, batch: dict[str, Any]) -> None:
        for _ in range(max(0, self.config.warmup)):
            self._forward_once(batch)
            self._sync_device()

    def _measure_latency(
        self,
        batch: dict[str, Any],
        repeat: int,
    ) -> tuple[float, float]:
        durations: list[float] = []
        loops = max(1, repeat)
        for _ in range(loops):
            start = time.perf_counter()
            self._forward_once(batch)
            self._sync_device()
            end = time.perf_counter()
            durations.append((end - start) * 1000.0)
        durations.sort()
        mean = sum(durations) / len(durations)
        idx = max(0, min(len(durations) - 1, int(math.ceil(len(durations) * 0.9)) - 1))
        p90 = durations[idx]
        return mean, p90

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
