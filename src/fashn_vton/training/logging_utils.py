"""
Logging for training runs: console + rotating file + optional TensorBoard,
plus system/GPU telemetry and a smoothed ETA.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..utils.logger import CustomFormatter


def setup_run_logger(name: str, log_dir: Path, filename: Optional[str] = None) -> logging.Logger:
    """Console (coloured) + file logger. Idempotent per name."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    sh = logging.StreamHandler()
    sh.setFormatter(CustomFormatter(timestamp=True))
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / (filename or f"{name}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


def gpu_stats() -> Dict[str, float]:
    """VRAM figures in GB. `reserved` is what actually counts against the card."""
    if not torch.cuda.is_available():
        return {}
    return {
        "vram_alloc_gb": torch.cuda.memory_allocated() / 1e9,
        "vram_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def system_stats() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        import psutil

        p = psutil.Process(os.getpid())
        out["ram_rss_gb"] = p.memory_info().rss / 1e9
        out["cpu_percent"] = p.cpu_percent(interval=None)
        out["ram_percent"] = psutil.virtual_memory().percent
    except Exception:
        pass
    try:  # GPU utilisation + temperature, useful because this card throttles (§16.1)
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            util, temp = r.stdout.strip().split("\n")[0].split(",")
            out["gpu_util_pct"] = float(util)
            out["gpu_temp_c"] = float(temp)
    except Exception:
        pass
    return out


class MetricTracker:
    """Running mean over a sliding window."""

    def __init__(self, window: int = 100):
        self.window = window
        self._v: Dict[str, deque] = {}

    def update(self, **kw: float) -> None:
        for k, v in kw.items():
            if v is None:
                continue
            self._v.setdefault(k, deque(maxlen=self.window)).append(float(v))

    def mean(self, k: str) -> Optional[float]:
        d = self._v.get(k)
        return sum(d) / len(d) if d else None

    def as_dict(self) -> Dict[str, float]:
        return {k: sum(d) / len(d) for k, d in self._v.items() if d}

    def reset(self) -> None:
        self._v.clear()


class ETA:
    """Smoothed time-remaining estimate. Uses a window so throttling is reflected."""

    def __init__(self, total_steps: int, window: int = 50):
        self.total = total_steps
        self.times: deque = deque(maxlen=window)
        self._last = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = now - self._last
        self._last = now
        self.times.append(dt)
        return dt

    @property
    def sec_per_step(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    def remaining(self, step: int) -> float:
        return max(0, self.total - step) * self.sec_per_step

    @staticmethod
    def fmt(seconds: float) -> str:
        seconds = int(max(0, seconds))
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        return f"{h:d}h{m:02d}m" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")


class TrainLogger:
    """Unified logging surface: console/file lines + TensorBoard scalars."""

    def __init__(self, cfg, run_dir: Path, total_steps: int, name: str = "train"):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_run_logger(name, self.run_dir, "train.log")
        self.tracker = MetricTracker()
        self.eta = ETA(total_steps)
        self.total_steps = total_steps
        self.writer = None
        if cfg.log.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.run_dir / "tb"))
                self.logger.info(f"TensorBoard -> {self.run_dir/'tb'}")
            except Exception as e:
                self.logger.warning(f"TensorBoard unavailable ({type(e).__name__}: {e}); file logging only")

    # ---- passthrough
    def info(self, m: str) -> None:
        self.logger.info(m)

    def warning(self, m: str) -> None:
        self.logger.warning(m)

    def error(self, m: str) -> None:
        self.logger.error(m)

    def scalars(self, step: int, **kw: float) -> None:
        if self.writer is None:
            return
        for k, v in kw.items():
            if v is not None:
                self.writer.add_scalar(k, v, step)

    def log_step(self, step: int, epoch: int, loss: float, lr: float, extra: Optional[Dict[str, Any]] = None) -> None:
        dt = self.eta.tick()
        self.tracker.update(loss=loss, step_time=dt)
        if step % self.cfg.log.log_every_steps:
            return

        g, s = gpu_stats(), (system_stats() if self.cfg.log.log_system_metrics else {})
        msg = (f"epoch {epoch} | step {step}/{self.total_steps} | "
               f"loss {loss:.4f} (avg {self.tracker.mean('loss'):.4f}) | "
               f"lr {lr:.2e} | {dt:.2f}s/it | eta {ETA.fmt(self.eta.remaining(step))}")
        if g:
            msg += f" | vram {g['vram_alloc_gb']:.2f}/{g['vram_peak_gb']:.2f}GB"
        if s.get("gpu_temp_c"):
            msg += f" | gpu {s.get('gpu_util_pct', 0):.0f}% {s['gpu_temp_c']:.0f}C"
        if s.get("ram_rss_gb"):
            msg += f" | ram {s['ram_rss_gb']:.1f}GB"
        if extra:
            msg += " | " + " ".join(f"{k} {v}" for k, v in extra.items())
        self.logger.info(msg)

        self.scalars(step, **{"train/loss": loss, "train/loss_avg": self.tracker.mean("loss"),
                              "train/lr": lr, "perf/sec_per_step": dt,
                              **{f"mem/{k}": v for k, v in g.items()},
                              **{f"sys/{k}": v for k, v in s.items()}})

    def log_validation(self, step: int, epoch: int, metrics: Dict[str, float]) -> None:
        parts = "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"[validation] epoch {epoch} step {step} | {parts}")
        self.scalars(step, **{f"val/{k}": v for k, v in metrics.items()})

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
