#!/usr/bin/env python3
"""
Build the Phase 5 hand-off dataset: ~2,000 verified saree training pairs.

Every retained sample is checked programmatically — file present, image decodes,
dimensions valid, no duplicate garment or person, frontal full-body standing pose,
and a genuine flat-lay saree as the garment input.

Output layout (VITON-HD convention, consumed directly by the training pipeline):

    <out>/
    ├── train/
    │   ├── image/<id>.jpg     person wearing the saree — also the ground truth
    │   ├── cloth/<id>.jpg     flat saree
    │   └── pairs.txt          "<id>.jpg <id>.jpg" per line
    ├── val/
    │   ├── image/<id>.jpg
    │   ├── cloth/<id>.jpg
    │   └── pairs.txt
    ├── metadata.csv           per-sample provenance and quality fields
    └── README.md

Usage:
    python scripts/build_handoff_dataset.py --out dataset_handoff --target 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 20260729


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def verify_image(path: Path, min_side: int) -> tuple[bool, str, int, int]:
    """Decode fully and check dimensions. Returns (ok, reason, w, h)."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            im.load()
    except Exception as e:
        return False, f"decode_failed:{type(e).__name__}", 0, 0
    if min(w, h) < min_side:
        return False, f"too_small:{w}x{h}", w, h
    if w < 1 or h < 1:
        return False, "invalid_dims", w, h
    return True, "", w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default="dataset", help="original dataset root")
    ap.add_argument("--clean-csv", default="data_clean/clean_all.csv")
    ap.add_argument("--out", default="dataset_handoff")
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--min-pose-score", type=float, default=1.1)
    ap.add_argument("--exclude-mannequin", action="store_true", help="keep only real-model subjects")
    args = ap.parse_args()

    src = Path(args.source_root)
    out = Path(args.out)
    df = pd.read_csv(args.clean_csv, dtype={"id": str})
    stages = []
    n0 = len(df)

    def step(name: str, mask) -> None:
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        stages.append({"stage": name, "removed": before - len(df), "remaining": len(df)})
        print(f"  {name:<42s} -{before - len(df):>5d}   remaining {len(df):>5d}")

    print(f"Building hand-off dataset from {n0} pre-cleaned pairs\n")
    print("Filtering")

    # The upstream cleaning cascade already removed: duplicates by garment hash,
    # low-resolution pairs, non-garment cloth images, on-model cloth photographs
    # and blouse-piece product renders. The gates below add pose and integrity.
    step(f"frontal full-body pose (score >= {args.min_pose_score})", df.pose_score >= args.min_pose_score)
    if args.exclude_mannequin:
        step("real-model subjects only", df.target_type == "model")

    # ---- integrity verification: every file decoded, not just stat()'d ----
    print("\nVerifying image integrity (decoding every file)")
    keep, reasons = [], []
    person_hashes, cloth_hashes = {}, {}
    dup_person = dup_cloth = 0

    for i, r in enumerate(df.itertuples(), 1):
        p_path, c_path = src / r.person_path, src / r.cloth_path
        if not p_path.exists() or not c_path.exists():
            keep.append(False), reasons.append("missing_file")
            continue
        ok_p, why_p, pw, ph = verify_image(p_path, args.min_side)
        if not ok_p:
            keep.append(False), reasons.append(f"person_{why_p}")
            continue
        ok_c, why_c, cw, ch = verify_image(c_path, args.min_side // 2)
        if not ok_c:
            keep.append(False), reasons.append(f"cloth_{why_c}")
            continue

        # byte-level duplicate detection across both sides of the pair
        hp, hc = file_sha1(p_path), file_sha1(c_path)
        if hp in person_hashes:
            keep.append(False), reasons.append(f"duplicate_person_of:{person_hashes[hp]}")
            dup_person += 1
            continue
        if hc in cloth_hashes:
            keep.append(False), reasons.append(f"duplicate_cloth_of:{cloth_hashes[hc]}")
            dup_cloth += 1
            continue
        person_hashes[hp], cloth_hashes[hc] = r.id, r.id
        keep.append(True), reasons.append("")
        if i % 500 == 0:
            print(f"  verified {i}/{len(df)}")

    df["_reason"] = reasons
    rejected = df[~np.array(keep)][["id", "_reason"]].copy()
    before = len(df)
    df = df[np.array(keep)].copy()
    stages.append(
        {
            "stage": "integrity + byte-duplicate verification",
            "removed": before - len(df),
            "remaining": len(df),
        }
    )
    print(
        f"  {'integrity + byte-duplicate verification':<42s} -{before - len(df):>5d}   "
        f"remaining {len(df):>5d}"
    )
    if dup_person or dup_cloth:
        print(f"    (duplicates: {dup_person} person, {dup_cloth} cloth)")

    # ---- trim to target, keeping the highest-quality samples ----
    if args.target and len(df) > args.target:
        df = df.sort_values(["garment_score", "person_mp"], ascending=False).head(args.target)
        stages.append(
            {
                "stage": f"trim to target ({args.target})",
                "removed": len(df) - args.target if len(df) > args.target else 0,
                "remaining": len(df),
            }
        )
        print(f"  {'trim to target by garment quality':<42s}        remaining {len(df):>5d}")

    # ---- deterministic split ----
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(df) * args.val_fraction)))
    df = df.iloc[idx].reset_index(drop=True)
    val_df = df.iloc[:n_val].sort_values("id").reset_index(drop=True)
    train_df = df.iloc[n_val:].sort_values("id").reset_index(drop=True)
    print(f"\nSplit: train {len(train_df)} · val {len(val_df)}")

    # ---- materialise ----
    if out.exists():
        shutil.rmtree(out)
    print("\nCopying files")
    manifest = []
    for split, part in (("train", train_df), ("val", val_df)):
        (out / split / "image").mkdir(parents=True, exist_ok=True)
        (out / split / "cloth").mkdir(parents=True, exist_ok=True)
        lines = []
        for r in part.itertuples():
            shutil.copy2(src / r.person_path, out / split / "image" / f"{r.id}.jpg")
            shutil.copy2(src / r.cloth_path, out / split / "cloth" / f"{r.id}.jpg")
            lines.append(f"{r.id}.jpg {r.id}.jpg")
            manifest.append(
                {
                    "id": r.id,
                    "split": split,
                    "source": r.source,
                    "title": r.title,
                    "url": r.url,
                    "target_type": r.target_type,
                    "person_w": r.person_w,
                    "person_h": r.person_h,
                    "cloth_w": r.cloth_w,
                    "cloth_h": r.cloth_h,
                    "pose_score": r.pose_score,
                    "garment_score": r.garment_score,
                }
            )
        (out / split / "pairs.txt").write_text("\n".join(lines) + "\n")
        print(f"  {split}: {len(part)} pairs")

    pd.DataFrame(manifest).to_csv(out / "metadata.csv", index=False)
    rejected.to_csv(out / "rejected.csv", index=False)

    summary = {
        "source_pairs": n0,
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "total_pairs": len(train_df) + len(val_df),
        "filters": stages,
        "rejection_reasons": rejected._reason.value_counts().head(20).to_dict(),
        "settings": {
            "min_side": args.min_side,
            "min_pose_score": args.min_pose_score,
            "exclude_mannequin": args.exclude_mannequin,
            "val_fraction": args.val_fraction,
            "seed": SEED,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nWrote {out}  ({size / 1e9:.2f} GB)")
    print(f"  train {len(train_df)} · val {len(val_df)} · total {len(train_df) + len(val_df)}")


if __name__ == "__main__":
    main()
