#!/usr/bin/env python3
"""
Generate inference samples from a series of training checkpoints.

Phase 4 Part 5: show the model's outputs progressing from the pretrained base through
early / middle / final checkpoints on the SAME inputs and SAME seed, so any visible
change is attributable to training rather than sampling noise.

    python eval/sample_progression.py \
        --checkpoints checkpoints/overfit/epoch0010.pt checkpoints/overfit/best.pt \
        --subset eval/overfit_subset.csv --n 4 --out outputs/phase4/progression
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image


def build_pipeline(weights_dir: str, height: int, width: int, aux_device: str = "cpu"):
    """TryOnPipeline with the Phase 2/3 4 GB fixes: aux on CPU, base staged via CPU."""
    from fashn_vton.pipeline import TryOnPipeline

    class _P(TryOnPipeline):
        def __init__(self, *a, aux_device="cpu", shape=(864, 576), **kw):
            self._aux, self._shape = aux_device, shape
            super().__init__(*a, **kw)

        def _setup_tryon_model(self):
            import os

            from fashn_vton.tryon_mmdit import TryOnModel
            from fashn_vton.utils import load_checkpoint

            self.tryon_model = TryOnModel(input_shape=self._shape)
            sd = load_checkpoint(os.path.join(self.weights_dir, "model.safetensors"), device="cpu")
            self.tryon_model.load_state_dict(sd)
            del sd
            gc.collect()
            self.tryon_model.to(self.device, dtype=self.inference_dtype).eval()

        def _setup_pose_model(self):
            import os

            from fashn_vton.dwpose import DWposeDetector

            d = "cuda:0" if self._aux == "cuda" else self._aux
            self.pose_model = DWposeDetector(
                checkpoints_dir=os.path.join(self.weights_dir, "dwpose"), device=d)

        def _setup_hp_model(self):
            from fashn_human_parser import FashnHumanParser

            self.hp_model = FashnHumanParser(device=self._aux)

    return _P(weights_dir=weights_dir, aux_device=aux_device, shape=(height, width))


def apply_lora(pipe, ckpt_path: str, logger=print):
    """Load adapters from a checkpoint and merge them into the base weights."""
    from fashn_vton.training.config import Config
    from fashn_vton.training.lora import inject_lora, load_lora_state_dict, merge_lora_into_base

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    inject_lora(pipe.tryon_model, cfg.lora)
    load_lora_state_dict(pipe.tryon_model, ck["lora"], strict=True)
    n = merge_lora_into_base(pipe.tryon_model)
    pipe.tryon_model.to(pipe.device, dtype=pipe.inference_dtype).eval()
    logger(f"  merged {n} LoRA modules (epoch {ck.get('epoch')}, step {ck.get('step')})")
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="eval/overfit_subset.csv")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--checkpoints", nargs="*", default=[],
                    help="checkpoints to sample, in training order")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--include-base", action="store_true", default=True)
    ap.add_argument("--n", type=int, default=4, help="samples per checkpoint")
    ap.add_argument("--height", type=int, default=648)
    ap.add_argument("--width", type=int, default=432)
    ap.add_argument("--category", default="one-pieces")
    ap.add_argument("--num-timesteps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--masked", action="store_true", default=True,
                    help="segmentation_free=False, the honest try-on setting")
    ap.add_argument("--out", default="outputs/phase4/progression")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(args.subset, dtype={"id": str}).head(args.n)
    root = Path(args.root)

    stages: List[tuple] = []
    if args.include_base:
        stages.append(("base", None))
    labels = args.labels or [Path(c).stem for c in args.checkpoints]
    stages += list(zip(labels, args.checkpoints))

    meta = {"stages": [s[0] for s in stages], "ids": list(sub.id),
            "height": args.height, "width": args.width, "seed": args.seed,
            "masked": args.masked, "num_timesteps": args.num_timesteps}

    for label, ckpt in stages:
        print(f"\n=== stage: {label} ===")
        t0 = time.time()
        pipe = build_pipeline(args.weights_dir, args.height, args.width)
        if ckpt:
            apply_lora(pipe, ckpt)

        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        for r in sub.itertuples():
            person = Image.open(root / r.person_path).convert("RGB")
            garment = Image.open(root / r.cloth_path).convert("RGB")
            res = pipe(person_image=person, garment_image=garment, category=args.category,
                       garment_photo_type="flat-lay", num_samples=1,
                       num_timesteps=args.num_timesteps, guidance_scale=args.guidance_scale,
                       seed=args.seed, segmentation_free=not args.masked)
            res.images[0].save(d / f"{r.id}.png")
            print(f"  {r.id} done")

        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  stage {label} took {time.time()-t0:.0f}s")

    (out / "progression.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
