"""
Pluggable evaluation metrics.

Phase 2 established that full-image SSIM/PSNR are close to blind to the failure that
actually matters: the sample that converted a saree into a kurta+palazzo scored the
run's HIGHEST SSIM (0.887), above every correctly-preserved saree (CONTEXT.md §16.3).
So this module is built as a **registry** rather than a fixed list — saree-specific
metrics can be dropped in without touching the evaluation driver, and pixel metrics are
retained for continuity with the Phase 2 baseline but must never be the sole criterion.

Adding a metric:

    @register("pallu_presence")
    class PalluPresence(Metric):
        higher_is_better = True
        def compute(self, pred, gt, ctx): ...

Every metric receives an optional `ctx` dict which may carry the agnostic mask, the
parse map, the garment image, or anything else a future metric needs.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Type

import numpy as np

_REGISTRY: Dict[str, Type["Metric"]] = {}


def register(name: str) -> Callable[[Type["Metric"]], Type["Metric"]]:
    def deco(cls: Type["Metric"]) -> Type["Metric"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def available() -> List[str]:
    return sorted(_REGISTRY)


def build(names: List[str], **kwargs) -> List["Metric"]:
    out = []
    for n in names:
        if n not in _REGISTRY:
            raise KeyError(f"unknown metric {n!r}; available: {available()}")
        out.append(_REGISTRY[n](**kwargs.get(n, {})))
    return out


class Metric:
    """Base class. `pred` and `gt` are uint8 HWC RGB arrays of equal shape."""

    name: str = "metric"
    higher_is_better: bool = True
    #: Metrics that only make sense over a whole set (e.g. FID) set this.
    is_distribution: bool = False
    #: Documented caveat surfaced in reports.
    caveat: str = ""

    def compute(self, pred: np.ndarray, gt: np.ndarray, ctx: Optional[Dict] = None) -> float:
        raise NotImplementedError

    def __call__(self, pred, gt, ctx=None) -> float:
        return float(self.compute(pred, gt, ctx))


# ----------------------------------------------------------------- pixel metrics
@register("ssim")
class SSIM(Metric):
    higher_is_better = True
    caveat = ("Near-blind to garment-category failure: CONTEXT.md §16.3 records a "
              "saree->kurta+palazzo conversion scoring the highest SSIM in the run.")

    def compute(self, pred, gt, ctx=None):
        from skimage.metrics import structural_similarity

        return structural_similarity(gt, pred, channel_axis=2, data_range=255)


@register("psnr")
class PSNR(Metric):
    higher_is_better = True
    caveat = "Dominated by background and skin; see SSIM caveat."

    def compute(self, pred, gt, ctx=None):
        from skimage.metrics import peak_signal_noise_ratio

        return peak_signal_noise_ratio(gt, pred, data_range=255)


@register("lpips")
class LPIPS(Metric):
    higher_is_better = False
    caveat = "Separates garment failures better than SSIM/PSNR, but distributions still overlap."

    _model = None
    _device = None

    def __init__(self, net: str = "alex", device: Optional[str] = None):
        self.net, self._want_device = net, device

    def _lazy(self):
        if LPIPS._model is None:
            import lpips as lpips_pkg
            import torch

            LPIPS._device = self._want_device or ("cuda" if torch.cuda.is_available() else "cpu")
            LPIPS._model = lpips_pkg.LPIPS(net=self.net).to(LPIPS._device).eval()
        return LPIPS._model

    def compute(self, pred, gt, ctx=None):
        import torch

        m = self._lazy()

        def t(a):
            x = torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            return x.to(LPIPS._device)

        with torch.no_grad():
            return m(t(pred), t(gt)).item()


# ------------------------------------------------------- masked-region variants
class _Masked(Metric):
    """
    Score only the region the model actually had to regenerate.

    This is the direct fix for the §16.3 problem: restricting to the agnostic mask
    removes the background/identity pixels that let a wrong garment score well.
    Requires `ctx['mask']` — a bool array, True where the model regenerated.
    """

    base: Optional[Metric] = None

    def compute(self, pred, gt, ctx=None):
        if not ctx or ctx.get("mask") is None:
            return float("nan")
        mask = ctx["mask"].astype(bool)
        if mask.sum() == 0:
            return float("nan")
        # Composite ground truth outside the mask so only the regenerated region differs.
        comp = gt.copy()
        comp[mask] = pred[mask]
        return self.base.compute(comp, gt, ctx)


@register("masked_ssim")
class MaskedSSIM(_Masked):
    higher_is_better = True
    caveat = "Requires ctx['mask']; returns NaN without it."

    def __init__(self):
        self.base = SSIM()


@register("masked_lpips")
class MaskedLPIPS(_Masked):
    higher_is_better = False
    caveat = "Requires ctx['mask']; returns NaN without it."

    def __init__(self, **kw):
        self.base = LPIPS(**kw)


# --------------------------------------------------------- saree-specific stubs
#
# Placeholders that define the interface Phase 4 will implement. They are registered
# but return NaN so a config can reference them today without breaking, and so the
# evaluation driver needs no change when they are filled in.

@register("saree_structure")
class SareeStructure(Metric):
    """
    Planned: does the output read as a draped saree rather than a stitched garment?

    Phase 2 FM-1 showed this is THE failure to measure, and that no pixel metric
    captures it. Intended implementation: a small classifier (saree / dress / two-piece)
    over generated outputs, reported as P(saree).
    """

    higher_is_better = True
    caveat = "NOT IMPLEMENTED — returns NaN. Phase 4 deliverable."

    def compute(self, pred, gt, ctx=None):
        return float("nan")


@register("pallu_presence")
class PalluPresence(Metric):
    """Planned: detect the shoulder drape, the most distinctive saree structure."""

    higher_is_better = True
    caveat = "NOT IMPLEMENTED — returns NaN. Phase 4 deliverable."

    def compute(self, pred, gt, ctx=None):
        return float("nan")


def describe(names: Optional[List[str]] = None) -> str:
    rows = []
    for n in (names or available()):
        cls = _REGISTRY[n]
        d = "higher" if cls.higher_is_better else "lower"
        rows.append(f"  {n:<18s} {d:<7s} better   {cls.caveat}")
    return "\n".join(rows)
