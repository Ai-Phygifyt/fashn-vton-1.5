"""
Evaluation driver: run a checkpoint (or the base model) over an eval subset, score it
with the pluggable metric registry, and optionally diff against a baseline run.

Designed so Phase 4 can compare fine-tuned against the Phase 2 baseline on the *same*
stratified subset and the *same* mode, which is the only comparison that means anything
(CONTEXT.md §13.3).

    # base model, reproduces the Phase 2 protocol
    python -m fashn_vton.evaluation.evaluate --subset eval/eval_subset.csv --out outputs/eval_base

    # a fine-tuned LoRA checkpoint
    python -m fashn_vton.evaluation.evaluate --subset eval/eval_subset.csv \
        --lora checkpoints/<run>/best.pt --out outputs/eval_ft

    # compare the two
    python -m fashn_vton.evaluation.evaluate --compare outputs/eval_base outputs/eval_ft
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .metrics import build as build_metrics
from .metrics import describe


def load_pipeline(weights_dir: str, lora_ckpt: Optional[str], device: Optional[str],
                  aux_device: str = "cpu", logger=None):
    """
    Build a TryOnPipeline, optionally with LoRA adapters applied.

    Mirrors the Phase 2 eval harness: auxiliary models on CPU and the base state dict
    staged through CPU, both required to fit a 4 GB card (CONTEXT.md §15.3).
    """
    import torch

    from ..pipeline import TryOnPipeline

    class _EvalPipeline(TryOnPipeline):
        def __init__(self, *a, aux_device="cpu", **kw):
            self._aux = aux_device
            super().__init__(*a, **kw)

        def _setup_tryon_model(self):
            import gc
            import os

            from ..tryon_mmdit import TryOnModel
            from ..utils import load_checkpoint

            self.tryon_model = TryOnModel()
            sd = load_checkpoint(os.path.join(self.weights_dir, "model.safetensors"), device="cpu")
            self.tryon_model.load_state_dict(sd)
            del sd
            gc.collect()
            self.tryon_model.to(self.device, dtype=self.inference_dtype).eval()

        def _setup_pose_model(self):
            import os

            from ..dwpose import DWposeDetector

            self.pose_model = DWposeDetector(
                checkpoints_dir=os.path.join(self.weights_dir, "dwpose"), device=self._aux)

        def _setup_hp_model(self):
            from fashn_human_parser import FashnHumanParser

            self.hp_model = FashnHumanParser(device=self._aux)

    pipe = _EvalPipeline(weights_dir=weights_dir, device=device, aux_device=aux_device)

    if lora_ckpt:
        from ..training.checkpoint import CheckpointManager  # noqa: F401
        from ..training.config import Config
        from ..training.lora import inject_lora, load_lora_state_dict, merge_lora_into_base

        ck = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(ck["config"])
        inject_lora(pipe.tryon_model, cfg.lora)
        load_lora_state_dict(pipe.tryon_model, ck["lora"], strict=True)
        # Merge so inference runs at full speed with no adapter overhead.
        n = merge_lora_into_base(pipe.tryon_model)
        pipe.tryon_model.to(pipe.device, dtype=pipe.inference_dtype).eval()
        if logger:
            logger.info(f"applied + merged {n} LoRA modules from {lora_ckpt} "
                        f"(epoch {ck.get('epoch')}, step {ck.get('step')})")
    return pipe


def run_eval(args) -> Path:
    import torch

    from ..training.logging_utils import setup_run_logger

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger("evaluate", out, "evaluate.log")
    logger.info("metrics:\n" + describe(args.metrics))

    sub = pd.read_csv(args.subset, dtype={"id": str})
    if args.limit:
        sub = sub.head(args.limit)
    root = Path(args.dataset_root)

    pipe = load_pipeline(args.weights_dir, args.lora, args.device, args.aux_device, logger)
    metrics = build_metrics(args.metrics)

    rows = []
    for i, r in enumerate(sub.itertuples(), 1):
        d = out / "samples" / r.id
        d.mkdir(parents=True, exist_ok=True)
        rec = {"id": r.id}
        for c in ("source", "fabric_family", "gscore_q", "res_bucket", "target_type", "is_clean"):
            if hasattr(r, c):
                rec[c] = getattr(r, c)
        try:
            person = Image.open(root / r.person_path).convert("RGB")
            garment = Image.open(root / r.cloth_path).convert("RGB")
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.time()
            res = pipe(person_image=person, garment_image=garment, category=args.category,
                       garment_photo_type=args.garment_photo_type, num_samples=1,
                       num_timesteps=args.num_timesteps, guidance_scale=args.guidance_scale,
                       seed=args.seed, segmentation_free=not args.masked)
            rec["seconds"] = round(time.time() - t0, 2)
            gen = res.images[0]

            shutil.copy2(root / r.person_path, d / "person.jpg")
            shutil.copy2(root / r.cloth_path, d / "garment.jpg")
            gen.save(d / "output.png")

            gt = Image.open(root / r.person_path).convert("RGB")
            if gt.size != gen.size:
                gt = gt.resize(gen.size, Image.LANCZOS)
            pred_a, gt_a = np.asarray(gen), np.asarray(gt)
            for m in metrics:
                try:
                    rec[m.name] = m(pred_a, gt_a, None)
                except Exception as e:
                    rec[m.name] = float("nan")
                    logger.warning(f"{m.name} failed on {r.id}: {type(e).__name__}: {e}")
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            logger.error(f"{r.id} failed: {rec['error']}")
        rows.append(rec)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        if i % 10 == 0:
            logger.info(f"  {i}/{len(sub)}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    cols = [m.name for m in metrics if m.name in df.columns]
    summary = {
        "n": int(len(df)),
        "ok": int(df.ok.sum()),
        "mode": "masked" if args.masked else "segfree",
        "lora": args.lora,
        "category": args.category,
        "num_timesteps": args.num_timesteps,
        "guidance_scale": args.guidance_scale,
        "means": {c: float(df[df.ok][c].mean()) for c in cols},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("summary:\n" + json.dumps(summary, indent=2))
    return out


def compare(a: str, b: str, out: Optional[str] = None) -> pd.DataFrame:
    """Diff two evaluation runs, joined per-sample so the comparison is paired."""
    from .metrics import _REGISTRY

    da = pd.read_csv(Path(a) / "metrics.csv", dtype={"id": str})
    db = pd.read_csv(Path(b) / "metrics.csv", dtype={"id": str})
    names = [c for c in da.columns if c in _REGISTRY and c in db.columns]
    j = da.merge(db, on="id", suffixes=("_a", "_b"))

    rows = []
    for n in names:
        ca, cb = f"{n}_a", f"{n}_b"
        if ca not in j or cb not in j:
            continue
        va, vb = j[ca].astype(float), j[cb].astype(float)
        if va.isna().all() or vb.isna().all():
            continue
        higher = _REGISTRY[n].higher_is_better
        delta = (vb - va).mean()
        rows.append({
            "metric": n,
            "a_mean": va.mean(),
            "b_mean": vb.mean(),
            "delta": delta,
            "improved": bool(delta > 0) == higher,
            "n_better": int(((vb > va) == higher).sum()),
            "n": int(len(j)),
            "caveat": _REGISTRY[n].caveat,
        })
    res = pd.DataFrame(rows)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
    return res


def make_boards(run_a: str, run_b: Optional[str], out_dir: str, limit: Optional[int] = None,
                label_a: str = "Base", label_b: str = "Fine-tuned", panel_h: int = 440) -> Path:
    """
    Comparison boards: Person | Garment | <run A> [| <run B>].

    With two runs this is the before/after view Phase 4 needs. Metrics are burned into
    the caption so a board is readable without cross-referencing a CSV — the Phase 2
    boards proved that matters when scanning a hundred samples.
    """
    from PIL import ImageDraw, ImageFont

    def font(sz, bold=False):
        for c in [f"/usr/share/fonts/TTF/DejaVuSans{'-Bold' if bold else ''}.ttf",
                  f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]:
            if Path(c).exists():
                try:
                    return ImageFont.truetype(c, sz)
                except Exception:
                    pass
        return ImageFont.load_default()

    f_lab, f_cap = font(15, True), font(13)
    a_dir = Path(run_a)
    b_dir = Path(run_b) if run_b else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ma = pd.read_csv(a_dir / "metrics.csv", dtype={"id": str}).set_index("id")
    mb = (pd.read_csv(b_dir / "metrics.csv", dtype={"id": str}).set_index("id")
          if b_dir else None)

    ids = [d.name for d in sorted((a_dir / "samples").iterdir()) if (d / "output.png").exists()]
    if limit:
        ids = ids[:limit]

    def fit(im):
        return im.resize((max(1, int(im.width * panel_h / im.height)), panel_h), Image.LANCZOS)

    for sid in ids:
        panels = [(fit(Image.open(a_dir / "samples" / sid / "person.jpg").convert("RGB")), "Person (input / GT)"),
                  (fit(Image.open(a_dir / "samples" / sid / "garment.jpg").convert("RGB")), "Garment (input)"),
                  (fit(Image.open(a_dir / "samples" / sid / "output.png").convert("RGB")), f"Output — {label_a}")]
        if b_dir and (b_dir / "samples" / sid / "output.png").exists():
            panels.append((fit(Image.open(b_dir / "samples" / sid / "output.png").convert("RGB")),
                           f"Output — {label_b}"))

        pad, lab_h, cap_h = 12, 26, 44
        Wd = sum(p.width for p, _ in panels) + pad * (len(panels) + 1)
        canvas = Image.new("RGB", (Wd, lab_h + panel_h + pad * 2 + cap_h), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        x = pad
        for im, lab in panels:
            d.text((x, 4), lab, font=f_lab, fill=(20, 20, 20))
            canvas.paste(im, (x, lab_h))
            x += im.width + pad

        y = lab_h + panel_h + pad
        meta = ma.loc[sid] if sid in ma.index else None
        if meta is not None:
            bits = [str(meta.get(k, "")) for k in ("source", "fabric_family", "target_type") if k in ma.columns]
            d.text((pad, y), f"{sid}   " + "   ".join(b for b in bits if b), font=f_cap, fill=(20, 20, 20))
        parts = []
        for lbl, tbl in ((label_a, ma), (label_b, mb)):
            if tbl is None or sid not in tbl.index:
                continue
            row = tbl.loc[sid]
            got = [f"{m} {float(row[m]):.3f}" for m in ("ssim", "psnr", "lpips")
                   if m in tbl.columns and pd.notna(row.get(m))]
            if got:
                parts.append(f"{lbl}: " + "  ".join(got))
        if parts:
            d.text((pad, y + 20), "   |   ".join(parts), font=f_cap, fill=(150, 30, 30))
        canvas.save(out / f"{sid}.jpg", quality=90)

    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate base or fine-tuned FASHN-VTON")
    ap.add_argument("--subset", default="eval/eval_subset.csv")
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--lora", default=None, help="path to a LoRA checkpoint")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--metrics", nargs="*", default=["ssim", "psnr", "lpips"])
    ap.add_argument("--category", default="one-pieces")
    ap.add_argument("--garment-photo-type", default="flat-lay", choices=["model", "flat-lay"])
    ap.add_argument("--masked", action="store_true",
                    help="segmentation_free=False — the honest try-on measurement")
    ap.add_argument("--num-timesteps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--aux-device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None)
    ap.add_argument("--boards", nargs="+", metavar="RUN", default=None,
                    help="build comparison boards from one or two eval run dirs")
    ap.add_argument("--boards-out", default="outputs/eval_boards")
    args = ap.parse_args()

    if args.compare:
        res = compare(*args.compare, out=None)
        print(res.to_string(index=False))
        return
    if args.boards:
        a = args.boards[0]
        b = args.boards[1] if len(args.boards) > 1 else None
        out = make_boards(a, b, args.boards_out, limit=args.limit)
        print(f"boards -> {out}")
        return
    run_eval(args)


if __name__ == "__main__":
    main()
