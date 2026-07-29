"""
Memory management for training on a 4 GB card.

Measured budget (CONTEXT.md §9, §16.1): 3.68 GiB usable, base weights 1.94 GB bf16,
inference peak 2.899 GB. LoRA training must fit gradients + optimiser state +
activations into what remains, which makes gradient checkpointing mandatory rather
than optional.
"""

from __future__ import annotations

import contextlib
import gc
import os
from typing import Callable, Optional

import torch


def configure_allocator(enable: bool = True) -> None:
    """
    Reduce fragmentation via expandable segments.

    Must run before the first CUDA allocation to take effect, so call it at process
    start. Measured to matter on this card, where headroom is ~0.8 GB.
    """
    if not enable:
        return
    cur = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in cur:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (cur + "," if cur else "") + "expandable_segments:True"


def resolve_dtype(precision: str) -> torch.dtype:
    """
    Map a precision name to a torch dtype.

    bf16 is strongly preferred over fp16 here: the checkpoint is stored in bf16, and
    bf16 needs no loss scaler, which removes a whole class of training instability.
    fp16 is a fallback for pre-Ampere cards.
    """
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def cleanup(deep: bool = False) -> None:
    """Release cached blocks. `deep` also runs a Python GC pass."""
    if deep:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextlib.contextmanager
def track_peak():
    """Context manager yielding a one-element list that receives the peak VRAM in GB."""
    reset_peak()
    out = [0.0]
    try:
        yield out
    finally:
        out[0] = peak_vram_gb()


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def oom_safe(fn: Callable, max_retries: int = 2, on_oom: Optional[Callable[[int], None]] = None):
    """
    Run `fn`, recovering from CUDA OOM by clearing the cache and retrying.

    A transient OOM on a card this small is usually fragmentation rather than a genuine
    capacity failure, so a retry after `empty_cache()` frequently succeeds. If every
    retry fails we re-raise — silently skipping steps would corrupt the training signal.
    """
    last: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not is_oom(e):
                raise
            last = e
            if on_oom:
                on_oom(attempt)
            cleanup(deep=True)
    raise last  # type: ignore[misc]


def enable_gradient_checkpointing(model: torch.nn.Module) -> int:
    """
    Turn on gradient checkpointing for the transformer blocks.

    Returns the number of block lists that were switched on. Requires the
    `gradient_checkpointing` support added to TryOnModel; raises if absent so a
    misconfiguration cannot silently blow the memory budget.
    """
    if not hasattr(model, "set_gradient_checkpointing"):
        raise AttributeError(
            "TryOnModel has no set_gradient_checkpointing(); the upstream patch adding "
            "checkpointing support is missing. Training will not fit in 4 GB without it."
        )
    return model.set_gradient_checkpointing(True)


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def estimate_training_memory(
    n_trainable: int, n_frozen: int, dtype: torch.dtype, optimizer: str = "adamw8bit"
) -> dict:
    """
    Static (non-activation) memory estimate in GB.

    Activations dominate the remainder and depend on resolution and checkpointing, so
    they are measured empirically rather than estimated here.
    """
    base_bytes = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    frozen = n_frozen * base_bytes
    params = n_trainable * 4  # LoRA params kept in fp32 for stability
    grads = n_trainable * 4
    per_state = 1 if optimizer == "adamw8bit" else 4
    opt = n_trainable * per_state * 2  # AdamW keeps two moments
    return {
        "frozen_weights_gb": frozen / 1e9,
        "trainable_params_gb": params / 1e9,
        "gradients_gb": grads / 1e9,
        "optimizer_state_gb": opt / 1e9,
        "static_total_gb": (frozen + params + grads + opt) / 1e9,
    }
