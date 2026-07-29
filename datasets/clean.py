"""
Garment-input cleaning cascade.

Phase 2 established that `dataset/*/cloth/` contains at least four kinds of input the
try-on model cannot use (CONTEXT.md §13.2, §16.4):

  1. on-model shots      19.2% of the eval subset — a second photo of a person wearing
                         the saree, not a flat garment
  2. captioned swatches  fabric close-ups with text burnt into the image
  3. technical drawings  line-art blouse illustrations, not photographs
  4. collages            several folded sarees in one frame (worst artifacts observed)

Critically, the dataset's own `garment_score` / `is_clean` flags do NOT separate these
— `is_clean=True` is 31% on-model and the top garment_score quartile is 53% on-model,
because the score asks "is a whole saree visible", which a lit model shot answers
*better* than a flat garment. So we cannot filter on the index alone.

What actually works
-------------------
Only category (1) is reliably detectable with the tools available. The human-parser
face-fraction test reproduces the Phase 2 measurement and is enabled by default.

Categories (2)-(4) are NOT reliably detectable by global image statistics. Measured
against 300 random cloth images (CONTEXT.md §21.2):

  * `flat_fraction > 0.70` matches 26% of the dataset — mostly legitimate flat-lays on
    plain studio backdrops — while the one known sketch sits at only ~p85. It detects
    "plain background", not "line drawing".
  * A bright-uniform-row "caption banner" test has p90 = 0.572 across the dataset,
    far above the known captioned swatch's 0.119. It detects "white backdrop".
  * Otsu thresholding merges the known collage into a single component, so a
    component count never fires.

Shipping filters that do not fire would give false confidence, so they are implemented
but DISABLED by default. Their statistics are still computed and persisted, because
they are exactly the features a small labelled classifier would need (see §21.2 for
the recommended follow-up).

This module runs an explicit cascade and logs the survivor count at every stage, then
writes `clean_train.csv` / `clean_validation.csv`.

Every image is decoded exactly once; all statistics are computed in that single pass.

Usage:
    python -m datasets.clean --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from fashn_human_parser import LABELS_TO_IDS
from PIL import Image

from training.config import Config

# Label ids are taken from the package constant, NEVER hardcoded.
#
# Phase 2 and the first Phase 3 pass used ids 11/12 for "face/hair", copied from the
# colour-table comment in scripts/debug_masks.py. That comment describes a different
# (stale) label scheme: in the installed parser 11/12 are *glasses* and *arms*. The
# on-model filter still worked — arms is a fine person-detector, and it actually catches
# mannequins, where no face or hair is detected — but the code did not measure what it
# claimed. See CONTEXT.md §26.2.
BODY_LABELS = ("face", "hair", "arms", "hands", "legs", "feet", "torso")
PERSON_IDS = tuple(LABELS_TO_IDS[k] for k in BODY_LABELS)

TOP_ID = LABELS_TO_IDS["top"]
DRESS_ID = LABELS_TO_IDS["dress"]
SKIRT_ID = LABELS_TO_IDS["skirt"]
PANTS_ID = LABELS_TO_IDS["pants"]


# ------------------------------------------------------------------ image statistics
def colorfulness(img: np.ndarray) -> float:
    """Hasler-Susstrunk colourfulness. Line drawings and greyscale diagrams score ~0."""
    b, g, r = img[..., 2].astype(np.float32), img[..., 1].astype(np.float32), img[..., 0].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def white_fraction(img: np.ndarray, thresh: int = 235) -> float:
    """Share of near-white pixels."""
    return float((img.min(axis=2) >= thresh).mean())


def flat_fraction(img: np.ndarray, win: int = 5, thresh: float = 3.0) -> float:
    """
    Share of pixels in low-local-variance (texture-free) regions.

    Intended as a sketch signal — a line drawing is flat almost everywhere while fabric
    has texture. Measured to be a poor discriminator on this dataset (26% of images
    exceed 0.70), so it is recorded but not used for filtering by default.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.blur(g, (win, win))
    m2 = cv2.blur(g * g, (win, win))
    local_std = np.sqrt(np.clip(m2 - m * m, 0, None))
    return float((local_std < thresh).mean())


def foreground_components(img: np.ndarray, min_area_frac: float) -> int:
    """
    Count large disconnected foreground blobs.

    A flat-lay saree is one connected garment. A collage of several folded sarees
    produces several large blobs, which is the signature we reject.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu against the (usually light) background, then clean up speckle.
    _, fg = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    total = img.shape[0] * img.shape[1]
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * total))


def probe_cloth(path: Path, parser, cfg, stat_size: int = 512, parse_size: int = 640) -> Dict:
    """Decode one cloth image and compute every statistic the cascade needs."""
    pil = Image.open(path).convert("RGB")
    small = pil.copy()
    small.thumbnail((stat_size, stat_size), Image.LANCZOS)
    bgr = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)

    out = {
        "colorfulness": round(colorfulness(bgr), 3),
        "white_frac": round(white_fraction(bgr), 4),
        "flat_frac": round(flat_fraction(bgr), 4),
        "n_components": foreground_components(bgr, cfg.cleaning.min_component_area_frac),
    }

    p = pil.copy()
    p.thumbnail((parse_size, parse_size), Image.LANCZOS)
    seg = parser.predict(np.array(p))
    tot = seg.size

    # Person evidence: any body label. Broader and better-named than the old
    # face+hair test, and it still fires on mannequins (which have no face/hair).
    out["person_frac"] = round(float(np.isin(seg, PERSON_IDS).sum()) / tot, 5)
    out["face_frac"] = round(
        float(np.isin(seg, [LABELS_TO_IDS["face"], LABELS_TO_IDS["hair"]]).sum()) / tot, 5
    )

    # Garment-shape evidence, used to detect blouse-piece renders (see §26.3).
    out["top_frac"] = round(float((seg == TOP_ID).sum()) / tot, 5)
    out["dress_frac"] = round(float((seg == DRESS_ID).sum()) / tot, 5)
    out["skirt_frac"] = round(float((seg == SKIRT_ID).sum()) / tot, 5)
    out["pants_frac"] = round(float((seg == PANTS_ID).sum()) / tot, 5)
    return out


def is_blouse_render(row, lo: float = 0.10, hi: float = 0.35) -> bool:
    """
    Detect a blouse-piece product render being supplied as the garment image.

    Indian saree listings routinely include the matching *blouse piece* as a separate
    photo — very often a templated mockup of a short, sleeved blouse on a plain
    background. Pairing that with a person wearing a full saree teaches the model the
    wrong garment entirely, and neither `garment_score` nor `is_clean` separates it.

    Signature: the parser sees ONLY a `top`, occupying a modest slice of the frame.
    Measured on hand-labelled examples, these renders cluster in a razor-thin band
    (top_frac 0.216-0.219) because they are the same mockup retextured per product.
    Real saree flat-lays instead show `dress`/`skirt`, or a much larger `top` (0.63).
    """
    return bool(
        row["dress_frac"] < 0.01
        and row["skirt_frac"] < 0.01
        and row["pants_frac"] < 0.01
        and lo < row["top_frac"] < hi
    )


# ------------------------------------------------------------------------- cascade
def build_clean_split(cfg: Config, stats: pd.DataFrame, logger=None) -> Dict[str, pd.DataFrame]:
    """Apply the cascade to a stats-augmented index and report survivors at each stage."""
    log = logger.info if logger else print
    c = cfg.cleaning
    df = stats.copy()
    stages = []

    def step(name: str, mask: pd.Series):
        nonlocal df
        before = len(df)
        df = df[mask.reindex(df.index, fill_value=False)]
        stages.append(dict(stage=name, removed=before - len(df), remaining=len(df)))
        log(f"  {name:<34s} -{before - len(df):>6d}   remaining {len(df):>6d}")

    log(f"Cleaning cascade — starting from {len(df)} pairs")
    if c.drop_duplicates:
        step("drop is_duplicate", ~df.is_duplicate.astype(bool))
    if c.require_hi_res:
        step("require is_hi_res", df.is_hi_res.astype(bool))
    step(f"garment_score > {c.min_garment_score}", df.garment_score > c.min_garment_score)
    step(f"reject on-model (person>{c.max_person_fraction})", df.person_frac < c.max_person_fraction)
    if c.reject_blouse_renders:
        step(
            "reject blouse-piece render",
            ~df.apply(lambda r: is_blouse_render(r, c.blouse_top_lo, c.blouse_top_hi), axis=1),
        )

    # Disabled by default — see the module docstring. Enabling either without first
    # validating new thresholds will remove large amounts of usable data.
    if c.enable_sketch_filter:
        log("  WARNING: sketch filter enabled; thresholds are UNVALIDATED (§21.2)")
        step(
            "reject sketch/diagram",
            ~((df.flat_frac > c.max_flat_fraction) | (df.colorfulness < c.min_colorfulness)),
        )
    if c.enable_collage_filter:
        log("  WARNING: collage filter enabled; thresholds are UNVALIDATED (§21.2)")
        step(
            f"reject collage (comp>{c.max_foreground_components})",
            df.n_components <= c.max_foreground_components,
        )

    # Deterministic train/val split. The dataset's own `test` split is reserved for the
    # Phase 2 benchmark, so validation is carved out of `train` only — never from test.
    train_pool = df[df.split == "train"]
    rng = np.random.RandomState(c.seed)
    idx = np.arange(len(train_pool))
    rng.shuffle(idx)
    n_val = max(1, int(len(train_pool) * c.val_fraction))
    val = train_pool.iloc[idx[:n_val]].sort_values("id")
    train = train_pool.iloc[idx[n_val:]].sort_values("id")

    log(
        f"\n  final: train {len(train)}  validation {len(val)}  "
        f"(held-out dataset test split untouched: {len(df[df.split == 'test'])} clean pairs available)"
    )
    return {"train": train, "val": val, "stages": pd.DataFrame(stages), "all_clean": df}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Build clean training/validation splits")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--stats-cache", default="data_clean/cloth_stats.parquet")
    ap.add_argument("--out-dir", default="data_clean")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    cfg.apply_overrides(args.overrides)

    from training.logging_utils import setup_run_logger

    logger = setup_run_logger("clean", Path(cfg.log.dir))
    root = Path(cfg.data.dataset_root)
    index = pd.read_parquet(cfg.data.index)
    if args.limit:
        index = index.head(args.limit)

    cache = Path(args.stats_cache)
    if cache.exists() and not args.recompute:
        stats = pd.read_parquet(cache)
        missing = set(index.id) - set(stats.id)
        logger.info(f"Loaded cached stats for {len(stats)} pairs ({len(missing)} missing)")
    else:
        stats, missing = pd.DataFrame(columns=["id"]), set(index.id)

    if missing:
        from fashn_human_parser import FashnHumanParser

        logger.info(f"Probing {len(missing)} cloth images on {args.device} …")
        parser = FashnHumanParser(device=args.device)
        todo = index[index.id.isin(missing)]
        rows, t0 = [], time.time()
        for i, r in enumerate(todo.itertuples(), 1):
            rec = {"id": r.id}
            try:
                rec.update(probe_cloth(root / r.cloth_path, parser, cfg))
                rec["probe_ok"] = True
            except Exception as e:  # corrupt/unreadable image -> excluded downstream
                rec.update(
                    colorfulness=0.0,
                    white_frac=1.0,
                    n_components=0,
                    face_frac=1.0,
                    limb_frac=1.0,
                    probe_ok=False,
                )
                logger.warning(f"probe failed for {r.id}: {type(e).__name__}: {e}")
            rows.append(rec)
            if i % 250 == 0:
                el = time.time() - t0
                logger.info(
                    f"  {i}/{len(todo)}  {el / i:.3f}s/img  eta {(len(todo) - i) * el / i / 60:.1f} min"
                )
        new = pd.DataFrame(rows)
        stats = new if stats.empty else pd.concat([stats, new], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        stats.to_parquet(cache, index=False)
        logger.info(f"Wrote stats cache -> {cache}")

    merged = index.merge(stats, on="id", how="inner")
    merged = merged[merged.probe_ok.astype(bool)]

    out = build_clean_split(cfg, merged, logger=logger)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    out["train"].to_csv(od / "clean_train.csv", index=False)
    out["val"].to_csv(od / "clean_validation.csv", index=False)
    out["stages"].to_csv(od / "cleaning_stages.csv", index=False)
    out["all_clean"].to_csv(od / "clean_all.csv", index=False)

    summary = {
        "input_pairs": int(len(merged)),
        "clean_pairs": int(len(out["all_clean"])),
        "train": int(len(out["train"])),
        "validation": int(len(out["val"])),
        "stages": out["stages"].to_dict("records"),
        "thresholds": cfg.cleaning.__dict__,
    }
    (od / "cleaning_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        f"\nWrote {od}/clean_train.csv ({len(out['train'])}) and clean_validation.csv ({len(out['val'])})"
    )


if __name__ == "__main__":
    main()
