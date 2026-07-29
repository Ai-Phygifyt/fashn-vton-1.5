"""
Training entrypoint for saree LoRA fine-tuning.

    python -m fashn_vton.training.train --config configs/default.yaml
    python -m fashn_vton.training.train --config configs/overfit.yaml --set train.epochs=200
    python -m fashn_vton.training.train --config configs/default.yaml --resume auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from .config import Config
from . import memory as mem


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FASHN-VTON saree LoRA fine-tuning")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="config overrides, e.g. --set optim.lr=5e-5 train.epochs=20")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="auto | none | <path> (overrides config)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and run a couple of steps, then exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    cfg.apply_overrides(args.overrides)
    if args.run_name:
        cfg.log.run_name = args.run_name
    if args.resume:
        cfg.checkpoint.resume = args.resume
    cfg.validate()

    # Must precede any CUDA allocation.
    mem.configure_allocator(cfg.memory.set_alloc_conf)

    import torch

    from .checkpoint import CheckpointManager
    from .data.dataset import build_dataloader
    from .engine import Trainer, build_model, build_optimizer, build_scheduler
    from .logging_utils import TrainLogger, setup_run_logger

    run_name = cfg.log.run_name or f"saree-lora-r{cfg.lora.rank}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg.log.dir) / run_name
    ckpt_dir = Path(cfg.checkpoint.dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    boot = setup_run_logger("setup", run_dir, "setup.log")
    boot.info(f"run: {run_name}")
    cfg.save(run_dir / "config.yaml")
    boot.info(f"config -> {run_dir/'config.yaml'}")

    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = mem.resolve_dtype(cfg.memory.precision)
    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        boot.warning("bf16 unsupported on this GPU; falling back to fp16")
        dtype = torch.float16
    boot.info(f"device={device} dtype={dtype} effective_batch={cfg.effective_batch_size}")

    # ---------------------------------------------------------------- data
    train_loader = build_dataloader(cfg, "train", logger=boot)
    val_loader = None
    if Path(cfg.data.val_csv).exists():
        try:
            val_loader = build_dataloader(cfg, "val", logger=boot)
        except Exception as e:
            boot.warning(f"validation set unavailable ({type(e).__name__}: {e}); training without it")

    steps_per_epoch = max(1, len(train_loader) // cfg.train.grad_accum_steps)
    total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
    boot.info(f"train batches {len(train_loader)} | steps/epoch {steps_per_epoch} | "
              f"total optimiser steps {total_steps}")

    # ---------------------------------------------------------------- model
    model, info = build_model(cfg, device, dtype, logger=boot)
    optimizer = build_optimizer(cfg, model, logger=boot)
    scheduler = build_scheduler(cfg, optimizer, total_steps)

    ckpt = CheckpointManager(cfg, ckpt_dir, logger=boot)
    start_epoch, resumed = 0, None
    resume_path = ckpt.resolve_resume()
    if resume_path:
        state = ckpt.load(resume_path, model, optimizer, scheduler)
        start_epoch, resumed = state["epoch"], state
    else:
        boot.info("starting from scratch (no resume checkpoint)")

    tlogger = TrainLogger(cfg, run_dir, total_steps)
    tlogger.info(f"LoRA modules {info['n_modules']} | trainable {info['trainable']:,} params")
    (run_dir / "model_info.json").write_text(json.dumps(
        {k: v for k, v in info.items() if k != "modules"}, indent=2, default=str))

    trainer = Trainer(cfg, model, optimizer, scheduler, device, dtype,
                      tlogger, ckpt, train_loader, val_loader)
    if resumed:
        trainer.step = resumed["step"]
        trainer.micro = resumed["step"] * cfg.train.grad_accum_steps

    if args.dry_run:
        cfg.train.max_steps = min(cfg.train.max_steps or 2, 2)
        tlogger.info("DRY RUN: 2 optimiser steps then exit")

    # ---------------------------------------------------------------- loop
    t0 = time.time()
    status = "completed"
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            tlogger.info(f"===== epoch {epoch+1}/{cfg.train.epochs} =====")
            stats = trainer.train_epoch(epoch, total_steps)

            metrics = trainer.validate() or {}
            if metrics:
                tlogger.log_validation(trainer.step, epoch, metrics)
            is_best = ckpt.update_best(metrics) if (metrics and cfg.checkpoint.save_best) else False

            if (epoch + 1) % max(1, cfg.checkpoint.save_every_epochs) == 0 or is_best:
                ckpt.save(model, optimizer, scheduler, epoch=epoch + 1, step=trainer.step,
                          metrics=metrics, tag=f"epoch{epoch+1:04d}.pt", is_best=is_best)

            tlogger.info(f"epoch {epoch+1} done in {stats['epoch_seconds']/60:.1f} min "
                         f"(peak vram {mem.peak_vram_gb():.2f} GB)")

            if cfg.train.max_steps and trainer.step >= cfg.train.max_steps:
                break
    except KeyboardInterrupt:
        status = "interrupted"
        tlogger.warning("interrupted — saving checkpoint so the run can resume")
        ckpt.save(model, optimizer, scheduler, epoch=trainer.epoch, step=trainer.step,
                  metrics={}, extra={"interrupted": True})
    except Exception:
        status = "failed"
        tlogger.error("training failed:\n" + traceback.format_exc())
        try:
            ckpt.save(model, optimizer, scheduler, epoch=trainer.epoch, step=trainer.step,
                      metrics={}, extra={"crashed": True})
            tlogger.info("emergency checkpoint saved")
        except Exception:
            tlogger.error("emergency checkpoint also failed")
        raise
    finally:
        elapsed = time.time() - t0
        summary = {
            "status": status,
            "run_name": run_name,
            # trainer.epoch is the 0-indexed epoch currently in flight, so a completed
            # run reports epoch+1; an interrupted one reports the epochs it finished.
            "epochs_completed": trainer.epoch + 1 if status == "completed" else trainer.epoch,
            "optimizer_steps": trainer.step,
            "elapsed_hours": round(elapsed / 3600, 3),
            "peak_vram_gb": round(mem.peak_vram_gb(), 3),
            "oom_events": trainer.oom_events,
            "skipped_batches": trainer.skipped_batches,
            "dataset_load_failures": getattr(train_loader.dataset, "n_failures", 0),
            "best_value": ckpt.best_value,
            "trainable_params": info["trainable"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        tlogger.info("=== training summary ===\n" + json.dumps(summary, indent=2, default=str))
        tlogger.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
