#!/usr/bin/env python3
"""
Detect whether each `cloth` image is a true flat-lay or actually an on-model shot.

DATASET.md describes `cloth/` as "flat saree", but inspection showed at least one
source (kalkifashion) supplying a second ON-MODEL photo as the cloth image. That
matters: TryOnPipeline's `garment_photo_type` switches both the garment pose
(dummy vs detected) and garment masking (off vs isolate-garment). Using
"flat-lay" on an on-model photo feeds the model a whole second person.

`garment_score` does NOT capture this — it is CLIP "is a whole saree visible",
which is high for both flat and worn shots.

Heuristic: run the human parser on the cloth image; if identity labels
(face/hair) plus limbs cover a meaningful share of pixels, a person is present.

Usage:
    python eval/probe_cloth_type.py --subset eval/eval_subset.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# fashn-human-parser label ids (see scripts/debug_masks.py)
FACE, HAIR, ARMS, HANDS, LEGS, FEET, TORSO = 11, 12, 13, 14, 15, 16, 17


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="eval/eval_subset.csv")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="eval/cloth_type.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--face-thresh", type=float, default=0.002,
                    help="min share of pixels labelled face+hair to call it on-model")
    args = ap.parse_args()

    from fashn_human_parser import FashnHumanParser

    hp = FashnHumanParser(device=args.device)
    root = Path(args.root)
    sub = pd.read_csv(args.subset, dtype={"id": str})

    rows = []
    for i, r in enumerate(sub.itertuples(), 1):
        img = Image.open(root / r.cloth_path).convert("RGB")
        # downscale for speed; parser is resolution-tolerant
        img.thumbnail((768, 768), Image.LANCZOS)
        seg = hp.predict(np.array(img))
        tot = seg.size
        face = float(np.isin(seg, [FACE, HAIR]).sum()) / tot
        limbs = float(np.isin(seg, [ARMS, HANDS, LEGS, FEET, TORSO]).sum()) / tot
        on_model = face >= args.face_thresh
        rows.append(dict(id=r.id, source=r.source, garment_score=r.garment_score,
                         gscore_q=r.gscore_q, is_clean=r.is_clean,
                         face_frac=round(face, 5), limb_frac=round(limbs, 5),
                         cloth_is_on_model=bool(on_model)))
        if i % 20 == 0:
            print(f"  {i}/{len(sub)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    n = len(df)
    k = int(df.cloth_is_on_model.sum())
    print(f"\nProbed {n} cloth images -> {args.out}")
    print(f"ON-MODEL (not flat-lay): {k}/{n}  ({100*k/n:.1f}%)\n")
    print("--- by source ---")
    print(df.groupby("source").agg(n=("id", "size"),
                                   on_model=("cloth_is_on_model", "sum"),
                                   pct=("cloth_is_on_model", lambda s: round(100*s.mean(), 1))).to_string())
    print("\n--- on-model rate by garment_score quartile ---")
    print(df.groupby("gscore_q").cloth_is_on_model.agg(["size", "sum", "mean"]).round(3).to_string())
    print("\n--- on-model rate by is_clean ---")
    print(df.groupby("is_clean").cloth_is_on_model.agg(["size", "sum", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
