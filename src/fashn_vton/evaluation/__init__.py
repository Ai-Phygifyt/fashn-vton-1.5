"""
Evaluation package.

Built as a metric registry so saree-specific measures can be plugged in later without
touching the driver — Phase 2 proved that SSIM/PSNR/LPIPS alone are misleading for this
task (CONTEXT.md §16.3, §17.4).
"""

from .metrics import Metric, available, build, describe, register

__all__ = ["Metric", "available", "build", "describe", "register"]
