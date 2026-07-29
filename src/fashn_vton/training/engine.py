"""
Training engine: model assembly, optimiser/scheduler construction, train and
validation loops.

Kept free of argument parsing and filesystem layout so it can be driven from
`train.py`, a notebook, or a test.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR

from ..tryon_mmdit import TryOnModel
from ..utils import load_checkpoint as load_base_checkpoint
from . import memory as mem
from .config import Config
from .losses import compute_loss
from .lora import inject_lora, lora_parameters, mark_only_lora_trainable, summarise


# ------------------------------------------------------------------------ model
def build_model(cfg: Config, device: torch.device, dtype: torch.dtype, logger=None
                ) -> Tuple[TryOnModel, Dict]:
    """
    Construct the model, load base weights, inject LoRA, freeze the base.

    Order matters: the base checkpoint must load BEFORE injection, because upstream
    `load_state_dict` is strict and knows nothing about adapter keys (CONTEXT.md §5.3).

    The state dict is staged through CPU: loading it straight to CUDA and then moving
    the model needs ~3.9 GB transiently and OOMs a 4 GB card (CONTEXT.md §15.3).
    """
    log = logger.info if logger else print

    model = TryOnModel(input_shape=(cfg.data.height, cfg.data.width))
    weights = Path(cfg.weights_dir) / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"base weights not found at {weights}. Run: python scripts/download_weights.py "
            f"--weights-dir {cfg.weights_dir}"
        )
    log(f"loading base weights from {weights} (staged via CPU)")
    sd = load_base_checkpoint(str(weights), device="cpu")
    model.load_state_dict(sd)
    del sd

    model, info = inject_lora(model, cfg.lora)
    n_trainable = mark_only_lora_trainable(model)
    log(summarise(model, info))

    # Base in the compute dtype; adapters stay fp32 (see lora.py).
    model.to(device=device, dtype=dtype)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    if cfg.memory.gradient_checkpointing:
        n = mem.enable_gradient_checkpointing(model)
        log(f"gradient checkpointing enabled on {n} block stacks")

    stats = mem.count_parameters(model)
    est = mem.estimate_training_memory(stats["trainable"], stats["frozen"], dtype, cfg.optim.optimizer)
    log(f"parameters: {stats['trainable']:,} trainable / {stats['total']:,} total "
        f"({stats['trainable_pct']:.3f}%)")
    log(f"estimated static VRAM: {est['static_total_gb']:.2f} GB "
        f"(frozen {est['frozen_weights_gb']:.2f} + params {est['trainable_params_gb']:.3f} "
        f"+ grads {est['gradients_gb']:.3f} + optim {est['optimizer_state_gb']:.3f})")

    info.update(stats)
    info["memory_estimate"] = est
    return model, info


# -------------------------------------------------------------------- optimiser
def build_optimizer(cfg: Config, model, logger=None):
    params = lora_parameters(model)
    if not params:
        raise RuntimeError("no trainable LoRA parameters found")
    kw = dict(lr=cfg.optim.lr, betas=tuple(cfg.optim.betas), eps=cfg.optim.eps,
              weight_decay=cfg.optim.weight_decay)

    if cfg.optim.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(params, **kw)
            if logger:
                logger.info("optimizer: bitsandbytes AdamW8bit")
            return opt
        except Exception as e:
            if logger:
                logger.warning(f"AdamW8bit unavailable ({type(e).__name__}: {e}); "
                               f"falling back to torch AdamW")
    opt = torch.optim.AdamW(params, **kw)
    if logger:
        logger.info("optimizer: torch AdamW")
    return opt


def build_scheduler(cfg: Config, optimizer, total_steps: int):
    warmup = max(0, cfg.optim.warmup_steps)
    floor = cfg.optim.min_lr_ratio
    kind = cfg.optim.scheduler

    def fn(step: int) -> float:
        if warmup and step < warmup:
            return step / max(1, warmup)
        if kind == "constant":
            return 1.0
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(max(prog, 0.0), 1.0)
        if kind == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
        if kind == "linear":
            return floor + (1 - floor) * (1 - prog)
        raise ValueError(f"unknown scheduler {kind!r}")

    return LambdaLR(optimizer, fn)


# ------------------------------------------------------------------------ loops
def move_batch(batch: Dict, device: torch.device, dtype: torch.dtype) -> Dict:
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif v.dtype.is_floating_point:
            out[k] = v.to(device, dtype=dtype, non_blocking=True)
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


class Trainer:
    """Owns the training state machine: accumulation, clipping, stepping, validation."""

    def __init__(self, cfg: Config, model, optimizer, scheduler, device, dtype,
                 tlogger, ckpt_manager, train_loader, val_loader=None):
        self.cfg = cfg
        self.model = model
        self.opt = optimizer
        self.sched = scheduler
        self.device = device
        self.dtype = dtype
        self.log = tlogger
        self.ckpt = ckpt_manager
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.step = 0
        self.epoch = 0
        self.micro = 0
        self.oom_events = 0
        self.skipped_batches = 0

        # fp16 needs a loss scaler; bf16 does not (that is a reason to prefer bf16).
        self.scaler = torch.amp.GradScaler("cuda") if dtype is torch.float16 else None
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(cfg.train.seed)

    # ---------------------------------------------------------------- one micro-step
    def _forward_backward(self, batch: Dict) -> float:
        out = compute_loss(self.model, batch, self.cfg, self.gen)
        loss = out["loss"] / self.cfg.train.grad_accum_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return float(out["loss"].detach())

    def train_micro_batch(self, batch: Dict) -> Optional[float]:
        """Forward+backward one micro-batch, recovering from transient OOM."""
        def run():
            return self._forward_backward(batch)

        def on_oom(attempt: int):
            self.oom_events += 1
            self.log.warning(f"CUDA OOM at step {self.step} (retry {attempt+1}); clearing cache")
            self.opt.zero_grad(set_to_none=True)

        if not self.cfg.memory.oom_retry:
            return run()
        try:
            return mem.oom_safe(run, self.cfg.memory.oom_max_retries, on_oom)
        except Exception as e:
            if mem.is_oom(e):
                self.skipped_batches += 1
                self.log.error(f"skipping batch after repeated OOM at step {self.step}")
                self.opt.zero_grad(set_to_none=True)
                mem.cleanup(deep=True)
                return None
            raise

    def optimizer_step(self) -> float:
        """Clip, step, schedule, zero. Returns the pre-clip grad norm."""
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.cfg.optim.max_grad_norm,
        )
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        return float(gnorm)

    # -------------------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total, n = 0.0, 0
        # Fixed generator so validation loss is comparable across epochs rather than
        # fluctuating with the noise/timestep draw.
        g = torch.Generator(device=self.device)
        g.manual_seed(self.cfg.train.seed + 12345)
        for i, batch in enumerate(self.val_loader):
            if max_batches and i >= max_batches:
                break
            batch = move_batch(batch, self.device, self.dtype)
            out = compute_loss(self.model, batch, self.cfg, g)
            total += float(out["loss"])
            n += 1
        self.model.train()
        mem.cleanup()
        return {"val_loss": total / max(1, n), "val_batches": n}

    # ------------------------------------------------------------------ epoch loop
    def train_epoch(self, epoch: int, total_steps: int) -> Dict[str, float]:
        self.epoch = epoch
        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        accum = self.cfg.train.grad_accum_steps
        running, count = 0.0, 0
        t_epoch = time.time()

        for i, raw in enumerate(self.train_loader):
            batch = move_batch(raw, self.device, self.dtype)
            loss = self.train_micro_batch(batch)
            if loss is None:
                continue
            running += loss
            count += 1
            self.micro += 1

            if self.micro % accum:
                continue

            gnorm = self.optimizer_step()
            self.step += 1
            lr = self.sched.get_last_lr()[0]
            self.log.log_step(self.step, epoch, running / max(1, count), lr,
                              extra={"gnorm": f"{gnorm:.3f}"} if gnorm else None)
            running, count = 0.0, 0

            if self.cfg.memory.empty_cache_every and self.step % self.cfg.memory.empty_cache_every == 0:
                mem.cleanup()
            if self.cfg.memory.gc_every and self.step % self.cfg.memory.gc_every == 0:
                mem.cleanup(deep=True)

            if (self.cfg.log.val_every_steps and self.step % self.cfg.log.val_every_steps == 0):
                m = self.validate()
                if m:
                    self.log.log_validation(self.step, epoch, m)
                    self._maybe_checkpoint(m, force_best_check=True)

            if (self.cfg.checkpoint.save_every_steps
                    and self.step % self.cfg.checkpoint.save_every_steps == 0):
                self.ckpt.save(self.model, self.opt, self.sched, epoch=epoch, step=self.step,
                               metrics={"train_loss": loss})

            if self.cfg.train.max_steps and self.step >= self.cfg.train.max_steps:
                self.log.info(f"reached max_steps={self.cfg.train.max_steps}")
                break

        return {"epoch_seconds": time.time() - t_epoch, "steps": self.step}

    def _maybe_checkpoint(self, metrics: Dict[str, float], force_best_check: bool = False) -> None:
        is_best = self.ckpt.update_best(metrics) if self.cfg.checkpoint.save_best else False
        if is_best or force_best_check:
            self.ckpt.save(self.model, self.opt, self.sched, epoch=self.epoch,
                           step=self.step, metrics=metrics, is_best=is_best)
