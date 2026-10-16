"""Periodic GPU memory reports during batched evaluation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

_GB = 1024 ** 3


def get_gpu_usage() -> Dict[str, Any]:
    """Memory statistics for every visible GPU, plus the Slurm job and node if set."""
    if not torch.cuda.is_available():
        return {"gpu_count": 0, "gpus": [], "process_info": {}}

    gpu_count = torch.cuda.device_count()
    gpus = []
    process_info = {}
    for gpu_id in range(gpu_count):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
            allocated_mem = torch.cuda.memory_allocated(gpu_id)
            reserved_mem = torch.cuda.memory_reserved(gpu_id)
            gpus.append({
                "id": gpu_id,
                "name": torch.cuda.get_device_properties(gpu_id).name,
                "total_mem_gb": total_mem / _GB,
                "free_mem_gb": free_mem / _GB,
                "used_mem_gb": (total_mem - free_mem) / _GB,
                "allocated_mem_gb": allocated_mem / _GB,
                "reserved_mem_gb": reserved_mem / _GB,
                "memory_utilization_pct": ((total_mem - free_mem) / total_mem) * 100,
            })
            if allocated_mem > 0:
                process_info[gpu_id] = {
                    "allocated_gb": allocated_mem / _GB,
                    "reserved_gb": reserved_mem / _GB,
                }
        except Exception as e:
            gpus.append({"id": gpu_id, "error": str(e)})

    return {
        "gpu_count": gpu_count,
        "gpus": gpus,
        "process_info": process_info,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "slurm_node": os.environ.get("SLURM_NODELIST", os.environ.get("SLURMD_NODENAME", "N/A")),
    }


def print_gpu_usage(batch_idx: Optional[int] = None, prefix: str = "") -> None:
    usage = get_gpu_usage()
    if usage["gpu_count"] == 0:
        print(f"{prefix}GPU Usage: No GPUs available")
        return

    header_parts = ["GPU Usage"]
    if batch_idx is not None:
        header_parts.append(f"(after batch {batch_idx})")
    if usage["slurm_job_id"] != "N/A":
        header_parts.append(f"[Job {usage['slurm_job_id']}, Node {usage['slurm_node']}]")

    rule = f"{prefix}{'─' * 80}"
    print(rule)
    print(f"{prefix}{' '.join(header_parts)}")
    print(rule)
    for gpu in usage["gpus"]:
        if "error" in gpu:
            print(f"{prefix}  GPU {gpu['id']}: ERROR - {gpu['error']}")
            continue
        print(f"{prefix}  GPU {gpu['id']}: {gpu['name']}")
        print(f"{prefix}    Memory: {gpu['used_mem_gb']:.2f} / {gpu['total_mem_gb']:.2f} GB "
              f"({gpu['memory_utilization_pct']:.1f}% used)")
        print(f"{prefix}    Free: {gpu['free_mem_gb']:.2f} GB")
        proc = usage["process_info"].get(gpu["id"])
        if proc:
            print(f"{prefix}    This process: {proc['allocated_gb']:.2f} GB allocated, "
                  f"{proc['reserved_gb']:.2f} GB reserved")
    print(rule)


class GPUMonitor:
    """Prints GPU memory usage every ``print_every_n_batches`` batches."""

    def __init__(self, print_every_n_batches: int = 5, prefix: str = "  "):
        self.print_every_n = print_every_n_batches
        self.prefix = prefix
        self.batch_count = 0
        self.enabled = torch.cuda.is_available()

    def record_batch(self, batch_idx: Optional[int] = None) -> None:
        if not self.enabled:
            return
        self.batch_count += 1
        if self.batch_count % self.print_every_n == 0:
            effective_idx = batch_idx if batch_idx is not None else self.batch_count
            print_gpu_usage(batch_idx=effective_idx, prefix=self.prefix)

    def print_final(self) -> None:
        if self.enabled:
            print_gpu_usage(prefix=self.prefix)
