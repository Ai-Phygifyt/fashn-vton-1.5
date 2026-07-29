"""
LoRA adapters for TryOnModel.

Full fine-tuning of the 971.8 M-parameter model needs ~17.5 GB (or ~5.8 GB even with
bf16 grads + 8-bit Adam) against a 3.68 GiB ceiling, so LoRA is the only viable local
strategy (CONTEXT.md §9.6). Base weights stay frozen in bf16; only the adapters train.

Design notes
------------
* Adapter weights are kept in **fp32** while the base runs in bf16. The adapters are a
  rounding error in memory terms (~11 M params at rank 16) and fp32 removes a source of
  optimiser noise. The LoRA branch casts to fp32 and back around its own matmuls.
* `B` is zero-initialised so the adapted model is **exactly** the base model at step 0.
  That makes "did training change anything?" a decidable question.
* Injection replaces `nn.Linear` in place, so `TryOnModel.forward` is untouched and the
  base checkpoint still loads strictly (CONTEXT.md §5.3 flagged strict loading as a
  blocker; we load the base first, then inject).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import LoRAConfig


class LoRALinear(nn.Module):
    """`nn.Linear` + a trainable low-rank update: y = W0·x + (alpha/r)·B·A·x."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0,
                 init_scale: float = 0.01):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Create adapters on the base layer's device. Training injects while the model is
        # still on CPU and moves afterwards, so this never mattered there — but the eval
        # path injects into an already-on-GPU model, and CPU adapters then produce
        # "Expected all tensors to be on the same device" at forward/merge time.
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32, device=dev))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming-style init on A; B stays zero so the delta is exactly zero at init.
        nn.init.normal_(self.lora_A, std=init_scale / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.lora_dropout(x).to(self.lora_A.dtype)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(h, self.lora_A), self.lora_B)
        return out + (delta * self.scaling).to(out.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """W0 + scaling·B·A — for exporting a standalone merged checkpoint."""
        w = self.base.weight.data
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return w + delta.to(device=w.device, dtype=w.dtype)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}")


def _leaf_name(qualified: str) -> str:
    """`double_blocks.3.img_mlp.0` -> `img_mlp.0`; `...img_attn.qkv` -> `qkv`."""
    parts = qualified.split(".")
    if parts[-1].isdigit() and len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


def _matches(qualified: str, targets: Iterable[str], include_blocks: List[str]) -> bool:
    if include_blocks and qualified.split(".")[0] not in include_blocks:
        return False
    return _leaf_name(qualified) in set(targets)


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> Tuple[nn.Module, Dict]:
    """
    Replace every matching `nn.Linear` with a `LoRALinear`.

    Call AFTER loading the base checkpoint (the base state dict has no adapter keys and
    `load_state_dict` is strict upstream).

    Returns (model, info) where info records what was adapted.
    """
    targets = cfg.resolve_targets()
    replaced: List[str] = []

    # Materialise the list first — we mutate the module tree while iterating.
    candidates = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name, mod in candidates:
        if not _matches(name, targets, cfg.include_blocks):
            continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        wrapped = LoRALinear(mod, cfg.rank, cfg.alpha, cfg.dropout, cfg.init_scale)
        setattr(parent, parts[-1], wrapped)
        replaced.append(name)

    if not replaced:
        raise RuntimeError(
            f"LoRA injection matched no modules. targets={targets} "
            f"include_blocks={cfg.include_blocks}. Check the preset against the model."
        )

    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    info = {
        "n_modules": len(replaced),
        "n_lora_params": n_lora,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "targets": targets,
        "include_blocks": cfg.include_blocks,
        "modules": replaced,
    }
    return model, info


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze everything except adapter parameters. Returns the trainable count."""
    n = 0
    for name, p in model.named_parameters():
        train = "lora_" in name
        p.requires_grad_(train)
        if train:
            n += p.numel()
    return n


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Only adapter tensors — a few tens of MB rather than 1.94 GB."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor], strict: bool = True) -> Dict:
    """Load adapter weights back into an already-injected model."""
    own = {n: p for n, p in model.named_parameters() if "lora_" in n}
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA state dict mismatch. missing={missing[:5]}... unexpected={unexpected[:5]}...\n"
            "This usually means rank/preset/include_blocks differ from the saved run."
        )
    with torch.no_grad():
        for k, p in own.items():
            if k in sd:
                if sd[k].shape != p.shape:
                    raise RuntimeError(f"shape mismatch for {k}: ckpt {tuple(sd[k].shape)} vs model {tuple(p.shape)}")
                p.copy_(sd[k].to(p.device, p.dtype))
    return {"missing": missing, "unexpected": unexpected}


@torch.no_grad()
def merge_lora_into_base(model: nn.Module) -> int:
    """
    Fold adapters into the base weights and restore plain `nn.Linear`.

    Produces a standalone checkpoint that the stock inference pipeline can load with no
    LoRA code present. Returns the number of modules merged.
    """
    merged = 0
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, mod in targets:
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        base = mod.base
        base.weight.data.copy_(mod.merged_weight())
        setattr(parent, parts[-1], base)
        merged += 1
    return merged


def summarise(model: nn.Module, info: Optional[Dict] = None) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        f"LoRA: {info['n_modules']} modules adapted (rank {info['rank']}, alpha {info['alpha']})" if info else "LoRA",
        f"  trainable {trainable:,} / {total:,} params  ({100*trainable/total:.3f}%)",
        f"  adapter fp32 size ~{trainable*4/1e6:.1f} MB   "
        f"optimizer(adamw8bit) ~{trainable*2/1e6:.1f} MB",
    ]
    return "\n".join(lines)
