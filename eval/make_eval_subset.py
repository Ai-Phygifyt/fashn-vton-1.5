#!/usr/bin/env python3
"""
Build a reproducible stratified evaluation subset from the saree dataset.

Population: the TEST split only (1,376 pairs). Rationale: Phase 3 will fine-tune
on the TRAIN split, so drawing the baseline from TEST keeps the benchmark
directly comparable before/after fine-tuning with zero contamination.

Stratification axes (all requested in the Phase 2 brief):
  1. source            — 12 e-commerce stores (primary stratum; strongest confound,
                         since each store has its own styling/photography)
  2. is_clean          — genuine flat-lay garment + hi-res + de-duped, vs not
  3. fabric_family     — derived from `title` (no explicit style column exists)
  4. resolution bucket — person image min-side
  5. garment_score     — quartile within the test split

Method: greedy multi-axis marginal matching. We set a target distribution for
every axis = its distribution in the test split, but floored so rare levels still
get enough samples to say anything about them (pure proportional sampling would
give e.g. ekaya 1 sample and the top garment-score quartile only ~8). We then
greedily add the pair that most reduces total absolute deviation from those
targets across all five axes at once, so no single axis is balanced at the
expense of another. Fully deterministic (fixed seed, stable hash tie-break).

Usage:
    python eval/make_eval_subset.py --n 120 --out eval/eval_subset.csv
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

SEED = 1337

# Precedence-ordered. Fabric *base* is listed before embellishment because drape
# behaviour (what a try-on model must get right) is driven by the base cloth.
FABRIC_FAMILIES = [
    ("sheer_flowy", ["georgette", "chiffon", "organza", " net ", "tissue", "crepe", "satin"]),
    ("traditional_silk", ["banarasi", "kanjivaram", "kanchipuram", "paithani", "patola", "tussar", "silk"]),
    ("cotton_handloom", ["cotton", "linen", "khadi", "chanderi", "jamdani", "handloom", "ikat",
                         "kalamkari", "maheshwari", "muslin", "bhagalpuri"]),
    ("printed", ["print", "floral", "bandhani", "lehriya", "dyed", "block"]),
    ("embellished", ["embroider", "sequin", "zardosi", "gota", "velvet", "zari", "mirror"]),
]


def assign_fabric(title: str) -> str:
    t = f" {str(title).lower()} "
    for name, kws in FABRIC_FAMILIES:
        if any(k in t for k in kws):
            return name
    return "other"


def res_bucket(min_side: int) -> str:
    if min_side < 512:
        return "<512"
    if min_side < 1024:
        return "512-1024"
    return ">=1024"


def stable_key(id_str: str) -> int:
    """Deterministic per-row shuffle key — stable across machines and pandas versions."""
    return int(hashlib.md5(f"{SEED}:{id_str}".encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--n", type=int, default=120, help="target subset size")
    ap.add_argument("--min-per-source", type=int, default=2)
    ap.add_argument("--out", default="eval/eval_subset.csv")
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_parquet(root / "index.parquet")
    te = df[df.split == "test"].copy()

    # Derived strata
    te["fabric_family"] = te.title.map(assign_fabric)
    te["min_side"] = te[["person_w", "person_h"]].min(axis=1)
    te["res_bucket"] = te.min_side.map(res_bucket)
    te["gscore_q"] = pd.qcut(te.garment_score, 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    te["_k"] = te.id.map(stable_key)

    # ---- Target marginals: test-split proportions, floored for rare levels ----
    AXES = ["source", "is_clean", "fabric_family", "res_bucket", "gscore_q", "target_type"]
    floor = args.min_per_source / args.n  # minimum share any level should get

    targets = {}
    for ax in AXES:
        p = te[ax].astype(str).value_counts(normalize=True)
        # available supply caps what we can ask for
        supply = te[ax].astype(str).value_counts() / args.n
        t = p.clip(lower=floor)
        t = t.clip(upper=supply)  # never target more than exists
        targets[ax] = (t / t.sum()).to_dict()

    # ---- Greedy multi-axis marginal matching ----
    cand = te.sort_values("_k", kind="mergesort").reset_index(drop=True)
    cols = {ax: cand[ax].astype(str).tolist() for ax in AXES}
    counts = {ax: dict.fromkeys(targets[ax], 0) for ax in AXES}
    chosen, taken = [], set()

    def deviation(counts_, n_):
        """Sum of |achieved share - target share| across every axis and level."""
        if n_ == 0:
            return sum(sum(t.values()) for t in targets.values())
        return sum(
            abs(counts_[ax].get(lv, 0) / n_ - tgt)
            for ax, t in targets.items()
            for lv, tgt in t.items()
        )

    n_target = min(args.n, len(cand))
    for step in range(n_target):
        best, best_dev = None, None
        for i in range(len(cand)):
            if i in taken:
                continue
            for ax in AXES:
                counts[ax][cols[ax][i]] = counts[ax].get(cols[ax][i], 0) + 1
            d = deviation(counts, step + 1)
            for ax in AXES:
                counts[ax][cols[ax][i]] -= 1
            if best_dev is None or d < best_dev - 1e-12:
                best, best_dev = i, d
        if best is None:
            break
        taken.add(best)
        chosen.append(best)
        for ax in AXES:
            counts[ax][cols[ax][best]] = counts[ax].get(cols[ax][best], 0) + 1

    sub = cand.iloc[sorted(chosen)].drop_duplicates("id").sort_values("id").reset_index(drop=True)
    sub = sub.drop(columns=["_k"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)

    # Manifest hash — proves reproducibility
    h = hashlib.sha256(",".join(sub.id).encode()).hexdigest()[:16]

    print(f"Selected {len(sub)} / {len(te)} test pairs   manifest_sha256[:16] = {h}")
    print(f"Written to {args.out}\n")
    for col in ["source", "is_clean", "fabric_family", "res_bucket", "gscore_q", "target_type"]:
        vc = sub[col].value_counts().sort_index()
        pop = te[col].value_counts(normalize=True).sort_index()
        print(f"--- {col} ---")
        for k, v in vc.items():
            print(f"    {str(k):18s} {v:4d}  ({100*v/len(sub):5.1f}%)   test-split: {100*pop.get(k,0):5.1f}%")
        print()


if __name__ == "__main__":
    main()
