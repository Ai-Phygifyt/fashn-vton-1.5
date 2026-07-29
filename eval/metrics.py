#!/usr/bin/env python3
"""
Baseline evaluation metrics for Phase 2.

IMPORTANT: the fashn-vton repository ships NO evaluation code of any kind — no
SSIM/PSNR/LPIPS/FID, no test set, no benchmark script (see CONTEXT.md §10.4).
Everything here is implemented by us for this project; nothing was "already
supported" to run.

Because our dataset is PAIRED (each person image shows that exact saree), the
person image doubles as ground truth and full-reference metrics are meaningful:

  SSIM  structural similarity      (higher better, 1.0 = identical)
  PSNR  peak signal-to-noise ratio (higher better, dB)
  LPIPS learned perceptual distance (LOWER better) — optional, needs `lpips`

Distribution metrics (FID/KID) are deliberately NOT computed: at n=120 FID is
dominated by estimator bias and would be actively misleading. Documented as
future work in CONTEXT.md.

Usage:
    python eval/metrics.py --out-dir outputs/phase2
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")


def load_pair(sample_dir: Path):
    """Load generated output and ground-truth person image, aligned to output size."""
    out = Image.open(sample_dir / "output.png").convert("RGB")
    gt = Image.open(sample_dir / "person.jpg").convert("RGB")
    if gt.size != out.size:
        gt = gt.resize(out.size, Image.LANCZOS)
    return np.asarray(out), np.asarray(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/phase2")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips as lpips_pkg
            import torch
            lpips_model = lpips_pkg.LPIPS(net="alex")
            lpips_dev = "cuda" if torch.cuda.is_available() else "cpu"
            lpips_model = lpips_model.to(lpips_dev).eval()
            print(f"LPIPS (AlexNet) loaded on {lpips_dev}")
        except Exception as e:
            print(f"LPIPS unavailable ({type(e).__name__}: {e}) — skipping, SSIM/PSNR only")
            lpips_model = None

    out_root = Path(args.out_dir)
    runlog = pd.read_csv(out_root / "runlog.csv", dtype={"id": str})

    rows = []
    for r in runlog.itertuples():
        if not r.ok:
            continue
        d = out_root / r.mode / r.id
        if not (d / "output.png").exists():
            continue
        out, gt = load_pair(d)
        rec = dict(id=r.id, mode=r.mode, source=r.source, is_clean=r.is_clean,
                   fabric_family=r.fabric_family, res_bucket=r.res_bucket,
                   gscore_q=r.gscore_q, target_type=r.target_type)
        rec["ssim"] = float(ssim_fn(gt, out, channel_axis=2, data_range=255))
        rec["psnr"] = float(psnr_fn(gt, out, data_range=255))
        if lpips_model is not None:
            import torch
            def t(a):
                x = torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
                return x.to(next(lpips_model.parameters()).device)
            with torch.no_grad():
                rec["lpips"] = float(lpips_model(t(out), t(gt)).item())
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "metrics.csv", index=False)
    mcols = [c for c in ["ssim", "psnr", "lpips"] if c in df.columns]

    print(f"\nScored {len(df)} generations -> {out_root/'metrics.csv'}\n")
    print("=== OVERALL (by mode) ===")
    print(df.groupby("mode")[mcols].agg(["mean", "std"]).round(4).to_string())

    for axis in ["source", "is_clean", "fabric_family", "res_bucket", "gscore_q", "target_type"]:
        print(f"\n=== by {axis} ===")
        print(df.groupby(["mode", axis])[mcols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
