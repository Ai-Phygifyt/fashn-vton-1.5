"""
Checkpointing and resume.

The repo ships load-only checkpoint support and no saving at all (CONTEXT.md §5.3), so
this is written from scratch.

Only LoRA adapters are persisted, not the 1.94 GB base — a checkpoint is a few tens of
MB, so keeping several costs little. The base weights are reloaded from `weights_dir`
on resume, which also guarantees a resumed run starts from exactly the same base.

A checkpoint captures everything needed to continue bit-for-bit:
  adapter weights, optimizer state, scheduler state, epoch/step counters, best metric,
  the full config, and RNG state for torch/cuda/numpy/python.

Writes are atomic (temp file + rename) so a crash mid-save cannot leave a corrupt
"latest" that breaks recovery.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import Config
from .lora import load_lora_state_dict, lora_state_dict

LATEST = "latest.pt"
BEST = "best.pt"


def _rng_state() -> Dict[str, Any]:
    s = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _restore_rng(s: Dict[str, Any]) -> None:
    if not s:
        return
    try:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch"].cpu() if torch.is_tensor(s["torch"]) else s["torch"])
        if torch.cuda.is_available() and "cuda" in s:
            torch.cuda.set_rng_state_all([t.cpu() if torch.is_tensor(t) else t for t in s["cuda"]])
    except Exception:
        # A restored run with fresh RNG is far better than a crashed one.
        pass


class CheckpointManager:
    def __init__(self, cfg: Config, run_dir: Path, logger=None):
        self.cfg = cfg
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.best_value: Optional[float] = None
        self._rotation: List[Path] = []

    def _log(self, m: str) -> None:
        if self.logger:
            self.logger.info(m)

    # ------------------------------------------------------------------- saving
    def save(self, model, optimizer, scheduler, *, epoch: int, step: int,
             metrics: Optional[Dict[str, float]] = None, extra: Optional[Dict] = None,
             tag: Optional[str] = None, is_best: bool = False) -> Path:
        payload = {
            "format_version": 1,
            "created": time.time(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "config": self.cfg.to_dict(),
            "lora": lora_state_dict(model),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": _rng_state(),
            "best_value": self.best_value,
            "extra": extra or {},
        }

        target = self.dir / (tag if tag else LATEST)
        self._atomic_save(payload, target)
        size_mb = target.stat().st_size / 1e6
        self._log(f"checkpoint saved: {target.name} (epoch {epoch}, step {step}, {size_mb:.1f} MB)")

        if tag and tag not in (LATEST, BEST):
            # A tagged (per-epoch) save must also refresh `latest.pt`, otherwise a run
            # whose save_every_steps never fires leaves no anchor and `resume: auto`
            # silently restarts from scratch.
            shutil.copyfile(target, self.dir / LATEST)
            self._rotate(target)
        if is_best:
            shutil.copyfile(target, self.dir / BEST)
            self._log(f"new best ({self.cfg.checkpoint.monitor}={metrics.get(self.cfg.checkpoint.monitor)}) "
                      f"-> {BEST}")
        return target

    @staticmethod
    def _atomic_save(payload: Dict, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on POSIX

    def _rotate(self, newest: Path) -> None:
        keep = self.cfg.checkpoint.keep_last
        if keep <= 0:
            return
        self._rotation.append(newest)
        while len(self._rotation) > keep:
            old = self._rotation.pop(0)
            if old.exists() and old.name not in (LATEST, BEST):
                old.unlink()
                self._log(f"pruned old checkpoint {old.name}")

    def update_best(self, metrics: Dict[str, float]) -> bool:
        """True if this is the best value seen for the monitored metric."""
        key, mode = self.cfg.checkpoint.monitor, self.cfg.checkpoint.monitor_mode
        if key not in metrics:
            return False
        v = metrics[key]
        if self.best_value is None or (v < self.best_value if mode == "min" else v > self.best_value):
            self.best_value = v
            return True
        return False

    # ------------------------------------------------------------------ loading
    def resolve_resume(self) -> Optional[Path]:
        spec = self.cfg.checkpoint.resume
        if spec in ("none", "", None):
            return None
        if spec == "auto":
            p = self.dir / LATEST
            return p if p.exists() else None
        p = Path(spec)
        if not p.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {p}")
        return p

    def load(self, path: Path, model, optimizer=None, scheduler=None,
             strict_lora: bool = True, restore_rng: bool = True) -> Dict:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        load_lora_state_dict(model, ck["lora"], strict=strict_lora)

        if optimizer is not None and ck.get("optimizer"):
            try:
                optimizer.load_state_dict(ck["optimizer"])
            except Exception as e:
                self._log(f"WARNING: optimizer state not restored ({type(e).__name__}: {e}); "
                          f"continuing with a fresh optimizer")
        if scheduler is not None and ck.get("scheduler"):
            try:
                scheduler.load_state_dict(ck["scheduler"])
            except Exception as e:
                self._log(f"WARNING: scheduler state not restored ({type(e).__name__}: {e})")

        if restore_rng:
            _restore_rng(ck.get("rng", {}))
        self.best_value = ck.get("best_value")

        self._log(f"resumed from {path.name}: epoch {ck['epoch']}, step {ck['step']}"
                  + (f", best {self.cfg.checkpoint.monitor}={self.best_value:.5f}"
                     if self.best_value is not None else ""))
        return {"epoch": ck["epoch"], "step": ck["step"], "metrics": ck.get("metrics", {}),
                "config": ck.get("config", {})}

    # ---------------------------------------------------------------- inspection
    @staticmethod
    def inspect(path: str | Path) -> Dict:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "epoch": ck.get("epoch"),
            "step": ck.get("step"),
            "metrics": ck.get("metrics"),
            "n_lora_tensors": len(ck.get("lora", {})),
            "n_lora_params": int(sum(v.numel() for v in ck.get("lora", {}).values())),
            "has_optimizer": ck.get("optimizer") is not None,
            "has_scheduler": ck.get("scheduler") is not None,
            "created": ck.get("created"),
        }


def export_merged_model(model, out_path: str | Path, logger=None) -> Path:
    """
    Merge adapters into the base weights and write a standalone `.safetensors`.

    The result loads in the stock `TryOnPipeline` with no training code present, which
    is how a fine-tuned model should ship.
    """
    from safetensors.torch import save_file

    from .lora import merge_lora_into_base

    n = merge_lora_into_base(model)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out))
    if logger:
        logger.info(f"merged {n} LoRA modules -> {out} ({out.stat().st_size/1e9:.2f} GB)")
    return out
