#!/usr/bin/env python3
"""
Validate a hand-off dataset before training.

Checks, per split:
  - every entry in pairs.txt resolves to files that exist
  - every image decodes and has valid dimensions
  - no orphan files (present on disk but absent from pairs.txt)
  - no duplicate filenames within a split, and no id overlap between splits
  - image/ and cloth/ id sets match exactly

Exits non-zero if any check fails, so it can gate a training run.

Usage:
    python scripts/validate_dataset.py --root dataset_handoff
    python scripts/validate_dataset.py --root dataset_handoff --quick   # skip decoding
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def validate_split(root: Path, split: str, quick: bool, min_side: int) -> tuple[int, list[str]]:
    errs: list[str] = []
    sd = root / split
    if not sd.is_dir():
        return 0, [f"{split}: directory missing"]

    pairs_file = sd / "pairs.txt"
    if not pairs_file.exists():
        return 0, [f"{split}: pairs.txt missing"]

    lines = [ln.strip() for ln in pairs_file.read_text().splitlines() if ln.strip()]
    seen: set[str] = set()
    referenced: set[str] = set()

    for i, ln in enumerate(lines, 1):
        parts = ln.split()
        if len(parts) != 2:
            errs.append(f"{split}:pairs.txt:{i}: expected 2 fields, got {len(parts)}")
            continue
        img_name, cloth_name = parts
        if img_name in seen:
            errs.append(f"{split}:pairs.txt:{i}: duplicate entry {img_name}")
        seen.add(img_name)
        referenced.add(Path(img_name).stem)

        for sub, name in (("image", img_name), ("cloth", cloth_name)):
            p = sd / sub / name
            if not p.exists():
                errs.append(f"{split}: missing file {sub}/{name}")
                continue
            if quick:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    w, h = im.convert("RGB").size
                if w < 1 or h < 1:
                    errs.append(f"{split}: invalid dimensions {sub}/{name} ({w}x{h})")
                elif min(w, h) < min_side:
                    errs.append(f"{split}: below min side {sub}/{name} ({w}x{h})")
            except Exception as e:
                errs.append(f"{split}: corrupt {sub}/{name} ({type(e).__name__})")

    # orphans and set agreement
    on_disk_img = {p.stem for p in (sd / "image").glob("*.jpg")}
    on_disk_cloth = {p.stem for p in (sd / "cloth").glob("*.jpg")}
    for orphan in sorted(on_disk_img - referenced):
        errs.append(f"{split}: image/{orphan}.jpg present but not listed in pairs.txt")
    for missing in sorted(on_disk_img ^ on_disk_cloth):
        errs.append(f"{split}: id {missing} present in only one of image/ cloth/")

    return len(lines), errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_handoff")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--min-side", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="skip full image decoding")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"dataset root not found: {root}")
        return 1

    total, all_errs, counts = 0, [], {}
    ids_by_split: dict[str, set[str]] = {}

    print(f"Validating {root}\n")
    for split in args.splits:
        n, errs = validate_split(root, split, args.quick, args.min_side)
        counts[split] = n
        total += n
        all_errs += errs
        status = "OK" if not errs else f"{len(errs)} problem(s)"
        print(f"  {split:<6s} {n:>6d} pairs   {status}")
        sd = root / split / "image"
        ids_by_split[split] = {p.stem for p in sd.glob("*.jpg")} if sd.is_dir() else set()

    # cross-split leakage
    splits = list(ids_by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = ids_by_split[splits[i]] & ids_by_split[splits[j]]
            if overlap:
                all_errs.append(f"leakage: {len(overlap)} ids shared between {splits[i]} and {splits[j]}")

    print(f"\n  total  {total:>6d} pairs")
    if all_errs:
        print(f"\nFAILED — {len(all_errs)} problem(s):")
        for e in all_errs[:40]:
            print(f"  - {e}")
        if len(all_errs) > 40:
            print(f"  … and {len(all_errs) - 40} more")
        return 1

    print("\nAll checks passed:")
    print("  - every pairs.txt entry resolves to existing files")
    print(
        "  - all images decode with valid dimensions"
        if not args.quick
        else "  - file existence verified (decoding skipped)"
    )
    print("  - no duplicate entries, no orphan files")
    print("  - image/ and cloth/ id sets agree")
    print("  - no id overlap between splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
