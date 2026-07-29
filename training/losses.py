"""
Rectified-flow training objective.

The repository ships no training code, so the objective is derived from the inference
contract (CONTEXT.md §6.2). `TryOnPipeline._sample` integrates `x <- x + dt·v` with
`t: 0 -> 1` starting from pure noise, so `t=0` is noise, `t=1` is data, and the model
predicts a velocity. Training therefore is:

    x0 ~ N(0, I)                      noise
    t  ~ p(t)                         timestep
    xt = (1 - t)·x0 + t·x1            linear interpolant
    v* = x1 - x0                      constant-velocity target
    L  = || model(xt, t, cond) - v* ||²

Timestep sampling matters. `get_rf_schedule(mu=1.5)` concentrates inference steps at
high noise, so `shifted` (the default) samples training timesteps from the same
density rather than uniformly — train and inference then agree about where model
capacity is spent.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.sampling import time_shift


def sample_timesteps(
    batch_size: int, cfg, device: torch.device, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Draw t in (0, 1). `t=1` is data, `t=0` is noise."""
    mode = cfg.train.timestep_sampling
    u = torch.rand(batch_size, device=device, generator=generator)

    if mode == "uniform":
        t = u
    elif mode == "shifted":
        # Same reparameterisation the inference schedule applies, so the training
        # timestep density matches where sampling actually spends its steps.
        t = time_shift(-cfg.train.time_shift_mu, 1.0, u.clamp(1e-5, 1 - 1e-5))
    elif mode == "logit_normal":
        n = torch.randn(batch_size, device=device, generator=generator)
        t = torch.sigmoid(cfg.train.logit_normal_mean + cfg.train.logit_normal_std * n)
    else:
        raise ValueError(f"unknown timestep_sampling {mode!r}")
    return t.clamp(1e-4, 1 - 1e-4)


def build_flow_batch(
    x1: torch.Tensor, cfg, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct the noised input and the velocity target.

    Returns (xt, t, v_target) with xt/v_target shaped like x1 and t shaped (B,).
    """
    b = x1.shape[0]
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = sample_timesteps(b, cfg, x1.device, generator).to(x1.dtype)
    t_ = t.view(b, *([1] * (x1.dim() - 1)))
    xt = (1.0 - t_) * x0 + t_ * x1
    v_target = x1 - x0
    return xt, t, v_target


def loss_weights(t: torch.Tensor, cfg) -> torch.Tensor:
    """Optional per-timestep weighting. `uniform` is the standard rectified-flow choice."""
    if cfg.train.loss_weighting == "uniform":
        return torch.ones_like(t)
    if cfg.train.loss_weighting == "snr":
        # For the linear interpolant, SNR = (t / (1-t))². Clamped to stay finite at the
        # endpoints; mildly emphasises low-noise steps where fine texture is resolved.
        snr = (t / (1.0 - t)).clamp(1e-3, 1e3) ** 2
        return (snr / (1.0 + snr)).to(t.dtype)
    raise ValueError(f"unknown loss_weighting {cfg.train.loss_weighting!r}")


def flow_matching_loss(v_pred: torch.Tensor, v_target: torch.Tensor, t: torch.Tensor, cfg) -> torch.Tensor:
    """
    Velocity regression loss.

    Computed in fp32 regardless of the autocast dtype: the squared error of a bf16
    tensor with ~3 decimal digits of mantissa is a poor training signal, and the cast
    is free relative to the forward pass.
    """
    vp, vt = v_pred.float(), v_target.float()
    if cfg.train.loss == "mse":
        per = F.mse_loss(vp, vt, reduction="none")
    elif cfg.train.loss == "huber":
        per = F.huber_loss(vp, vt, reduction="none", delta=cfg.train.huber_delta)
    else:
        raise ValueError(f"unknown loss {cfg.train.loss!r}")

    per = per.flatten(1).mean(1)  # (B,)
    w = loss_weights(t.float(), cfg)
    return (per * w).mean()


def make_cond_dropout_mask(
    batch_size: int, prob: float, device: torch.device, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Bernoulli keep-mask for classifier-free guidance training.

    `TryOnModel.forward` already accepts a `mask` and routes it through
    `apply_conditional_dropout` for ca_images, poses, garment and the class label — the
    training hook exists upstream (CONTEXT.md §6.2). True == keep the conditioning.

    Without this the unconditional branch is never trained and CFG at inference
    degrades, so `prob` should stay non-zero.
    """
    if prob <= 0:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    r = torch.rand(batch_size, device=device, generator=generator)
    return r >= prob


def compute_loss(
    model, batch: Dict[str, torch.Tensor], cfg, generator: Optional[torch.Generator] = None
) -> Dict[str, torch.Tensor]:
    """
    One full training forward pass.

    `batch` must provide: person (target x1), ca_image, garment, person_pose,
    garment_pose, category.
    """
    x1 = batch["person"]
    xt, t, v_target = build_flow_batch(x1, cfg, generator)
    mask = make_cond_dropout_mask(x1.shape[0], cfg.train.cond_dropout_prob, x1.device, generator)

    v_pred = model(
        xt,
        t,
        ca_images=batch["ca_image"],
        garment_images=batch["garment"],
        person_poses=batch["person_pose"],
        garment_poses=batch["garment_pose"],
        garment_categories=batch["category"],
        mask=mask,
    )["x"]

    loss = flow_matching_loss(v_pred, v_target, t, cfg)
    return {"loss": loss, "t_mean": t.detach().float().mean(), "cond_kept": mask.float().mean()}
