#!/usr/bin/env python3
"""
Select and FREEZE the Phase 4 overfit subset.

Purpose is debugging, not generalisation: we need a handful of samples the pipeline
should be able to memorise. Quality matters far more than representativeness here — a
bad garment input would make a failure ambiguous ("is the pipeline broken, or is the
data wrong?"), which defeats the entire point of an overfit test.

Criteria (in order of application):
  * from `clean_train.csv` only  -> already de-duplicated, hi-res, garment_score > 0,
                                    and on-model cloth images removed (CONTEXT.md §25.1)
  * real models only             -> mannequins are out-of-distribution (Phase 2 §16.2)
  * top-half garment_score       -> strongest genuine flat garments
  * large person images          -> avoids upscaling artifacts
  * diverse colour + fabric      -> greedy pick maximising spread so a visual check can
                                    tell "learned the dataset" from "learned one colour"

Dominant colour is measured from the actual garment pixels rather than parsed from the
title, because titles are unreliable ("Rani Pink Banarasi Georgette").

Writes `eval/overfit_subset.csv` plus a manifest hash. Frozen: re-running reproduces
the identical set, and Phase 5 debugging can always come back to it.

    python eval/make_overfit_subset.py --n 24
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 4242

# Rejected by eye after the automated cascade passed them. Recorded individually so the
# frozen subset is auditable and the residual failure modes are visible rather than
# quietly dropped. Each is a case no current filter detects (CONTEXT.md §26.4).
MANUAL_EXCLUSIONS = {
    "0000954": "cloth is a blouse-fabric swatch captioned 'Blouse Piece'",
    "0004366": "cloth is a blouse-fabric swatch captioned 'blouse piece'",
    "0005714": "cloth is an angled blouse render — outside the frontal-template top_frac band",
    "0009538": "PERSON image is a Libas size chart illustration, not a photograph",
    "0006281": "cloth is a white blouse render — angled/embellished variant the filter misses",
}

FABRIC_FAMILIES = [
    ("sheer_flowy", ["georgette", "chiffon", "organza", " net ", "tissue", "crepe", "satin"]),
    ("traditional_silk", ["banarasi", "kanjivaram", "kanchipuram", "paithani", "patola", "tussar", "silk"]),
    ("cotton_handloom", ["cotton", "linen", "khadi", "chanderi", "jamdani", "handloom", "ikat",
                         "kalamkari", "maheshwari", "muslin", "bhagalpuri"]),
    ("printed", ["print", "floral", "bandhani", "lehriya", "dyed", "block"]),
    ("embellished", ["embroider", "sequin", "zardosi", "gota", "velvet", "zari", "mirror"]),
]


def fabric_of(title: str) -> str:
    t = f" {str(title).lower()} "
    for name, kws in FABRIC_FAMILIES:
        if any(k in t for k in kws):
            return name
    return "other"


def dominant_hue(path: Path) -> tuple[float, float, float]:
    """(hue 0-1, saturation, value) of the garment's dominant colour."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((128, 128), Image.LANCZOS)
    a = np.asarray(im).reshape(-1, 3) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in a])
    # ignore near-white/near-black background pixels when picking the hue
    keep = (hsv[:, 1] > 0.15) & (hsv[:, 2] > 0.15) & (hsv[:, 2] < 0.97)
    sel = hsv[keep] if keep.sum() > 50 else hsv
    hist, edges = np.histogram(sel[:, 0], bins=18, range=(0, 1))
    h = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
    return h, float(sel[:, 1].mean()), float(sel[:, 2].mean())


def hue_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data_clean/clean_train.csv")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pool", type=int, default=400, help="candidates to colour-profile")
    ap.add_argument("--out", default="eval/overfit_subset.csv")
    ap.add_argument("--exclude", nargs="*", default=list(MANUAL_EXCLUSIONS),
                    help="ids rejected by visual inspection (see MANUAL_EXCLUSIONS)")
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(args.train_csv, dtype={"id": str})
    n0 = len(df)

    if args.exclude:
        df = df[~df.id.isin(set(args.exclude))]
    df = df[df.target_type == "model"]
    n1 = len(df)
    df = df[df.garment_score >= df.garment_score.median()]
    n2 = len(df)
    df = df[df[["person_w", "person_h"]].min(axis=1) >= 768]
    n3 = len(df)
    df["fabric_family"] = df.title.map(fabric_of)

    print(f"pool: {n0} clean_train -> {n1} real-model -> {n2} top-half garment_score "
          f"-> {n3} person min-side >= 768px")

    # Deterministic candidate pool, spread across fabrics and sources.
    rng = np.random.RandomState(SEED)
    df = df.assign(_k=[int(hashlib.md5(f"{SEED}:{i}".encode()).hexdigest()[:8], 16) for i in df.id])
    df = df.sort_values("_k").reset_index(drop=True)
    pool = df.groupby("fabric_family", group_keys=False).head(max(20, args.pool // 5)).head(args.pool)
    print(f"profiling colour for {len(pool)} candidates …")

    rows = []
    for r in pool.itertuples():
        try:
            h, s, v = dominant_hue(root / r.cloth_path)
            rows.append({"id": r.id, "hue": h, "sat": s, "val": v})
        except Exception:
            pass
    prof = pd.DataFrame(rows)
    pool = pool.merge(prof, on="id", how="inner")

    # Greedy max-min: repeatedly take the candidate furthest from everything chosen so
    # far, in (hue, fabric, source) space. Guarantees visible variety in the boards.
    chosen: list[int] = []
    remaining = list(range(len(pool)))
    # seed with the most saturated sample so the first pick is a vivid, easy-to-read one
    first = int(pool.sat.idxmax())
    chosen.append(first)
    remaining.remove(first)

    while len(chosen) < min(args.n, len(pool)) and remaining:
        best, best_score = None, -1.0
        for i in remaining:
            ri = pool.iloc[i]
            d = min(
                hue_dist(ri.hue, pool.iloc[c].hue)
                + (0.35 if ri.fabric_family != pool.iloc[c].fabric_family else 0.0)
                + (0.15 if ri.source != pool.iloc[c].source else 0.0)
                for c in chosen
            )
            if d > best_score:
                best, best_score = i, d
        chosen.append(best)
        remaining.remove(best)

    sub = pool.iloc[sorted(chosen)].drop(columns=["_k"]).sort_values("id").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)

    manifest = hashlib.sha256(",".join(sub.id).encode()).hexdigest()[:16]
    print(f"\nSelected {len(sub)} samples -> {args.out}")
    print(f"FROZEN manifest sha256[:16] = {manifest}\n")
    print("fabric:", sub.fabric_family.value_counts().to_dict())
    print("source:", sub.source.value_counts().to_dict())
    print(f"hue spread: {sub.hue.min():.2f}-{sub.hue.max():.2f} over {sub.hue.nunique()} distinct bins")
    print(f"garment_score: {sub.garment_score.min():.4f}-{sub.garment_score.max():.4f}")
    print(f"person min-side: {sub[['person_w','person_h']].min(axis=1).min()}-"
          f"{sub[['person_w','person_h']].min(axis=1).max()} px")
    print("\nselected ids:", " ".join(sub.id))


if __name__ == "__main__":
    main()
