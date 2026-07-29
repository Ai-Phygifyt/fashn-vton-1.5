"""
Offline preprocessing cache.

DWPose (YOLOX-L + pose net) and the human parser together cost ~0.6 GB of VRAM plus an
ONNX Runtime arena, and running them inside the training loop would both compete for
the 3.68 GiB budget and dominate step time. Phase 1 §9.5/§10.2 flagged caching as a
prerequisite for training on this card; this module implements it.

For every sample we cache, at the model's canonical geometry (the exact
`pre_resize` -> `ResizePad` chain `TryOnPipeline.__call__` applies, so training inputs
are pixel-identical to inference inputs):

    person/<id>.jpg        resized+padded person image      (RGB)
    garment/<id>.jpg       resized+padded garment image     (RGB)
    person_pose/<id>.png   DWPose skeleton, grayscale       (L)
    garment_pose/<id>.png  DWPose skeleton or dummy         (L)
    parse/<id>.png         human-parser label map, uint8    (L)

Caching the resized RGB as well removes a 1400x2096 JPEG decode from every epoch, which
is the single largest data-loading cost.

The pass is resumable: existing complete sample sets are skipped.

Usage:
    python -m datasets.preprocess --config configs/default.yaml --split train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from models.preprocessing import AspectPreserveResize, ResizePad
from training.config import Config
from utils import get_dummy_dw_keypoints

SUBDIRS = ("person", "garment", "person_pose", "garment_pose", "parse")


def cache_paths(cache_dir: Path, sid: str) -> Dict[str, Path]:
    return {
        "person": cache_dir / "person" / f"{sid}.jpg",
        "garment": cache_dir / "garment" / f"{sid}.jpg",
        "person_pose": cache_dir / "person_pose" / f"{sid}.png",
        "garment_pose": cache_dir / "garment_pose" / f"{sid}.png",
        "parse": cache_dir / "parse" / f"{sid}.png",
    }


def is_cached(cache_dir: Path, sid: str) -> bool:
    return all(p.exists() for p in cache_paths(cache_dir, sid).values())


class Preprocessor:
    """Owns the pose + parsing models and the geometry transforms."""

    def __init__(self, cfg: Config, device: str = "cpu", weights_dir: Optional[str] = None):
        from fashn_human_parser import FashnHumanParser

        from models.dwpose import DWposeDetector, draw_pose

        self.cfg = cfg
        self.draw_pose = draw_pose
        h, w = cfg.data.height, cfg.data.width
        max_dim = max(h, w)
        # Identical to TryOnPipeline.__init__
        self.pre_resize = AspectPreserveResize(target_size=(max_dim, max_dim), mode="fit", backend="pil")
        self.resize_pad = ResizePad((w, h), backend="opencv")

        wd = Path(weights_dir or cfg.weights_dir)
        # DWPose parses an ordinal out of the device string, so bare "cuda" crashes it;
        # the human parser wants the plain form. Normalise both.
        dw_device = "cuda:0" if device == "cuda" else device
        hp_device = device.split(":")[0] if device.startswith("cuda") else device
        self.pose = DWposeDetector(checkpoints_dir=str(wd / "dwpose"), device=dw_device)
        self.parser = FashnHumanParser(device=hp_device)

    def process(
        self, person_path: Path, garment_path: Path, garment_is_flat: bool = True
    ) -> Dict[str, np.ndarray]:
        person = self.pre_resize(Image.open(person_path).convert("RGB"), allow_upsampling=False)
        garment = self.pre_resize(Image.open(garment_path).convert("RGB"), allow_upsampling=False)
        p_np, g_np = np.array(person), np.array(garment)

        # DWPose expects BGR
        p_pose = self.pose(p_np[..., ::-1])
        g_pose = get_dummy_dw_keypoints() if garment_is_flat else self.pose(g_np[..., ::-1])
        p_pose_img = self.draw_pose(p_pose, p_np.shape[0], p_np.shape[1], grayscale=True)
        g_pose_img = self.draw_pose(g_pose, g_np.shape[0], g_np.shape[1], grayscale=True)

        parse = self.parser.predict(p_np)

        return {
            "person": self.resize_pad(p_np),
            "garment": self.resize_pad(g_np),
            "person_pose": self.resize_pad(p_pose_img, interpolation=cv2.INTER_NEAREST_EXACT),
            "garment_pose": self.resize_pad(g_pose_img, interpolation=cv2.INTER_NEAREST_EXACT),
            # Nearest is mandatory: these are class ids, not intensities.
            "parse": self.resize_pad(parse.astype(np.uint8), interpolation=cv2.INTER_NEAREST_EXACT),
        }


def write_sample(cache_dir: Path, sid: str, arrays: Dict[str, np.ndarray]) -> None:
    paths = cache_paths(cache_dir, sid)
    Image.fromarray(arrays["person"]).save(paths["person"], quality=95, subsampling=0)
    Image.fromarray(arrays["garment"]).save(paths["garment"], quality=95, subsampling=0)
    Image.fromarray(arrays["person_pose"]).save(paths["person_pose"], optimize=True)
    Image.fromarray(arrays["garment_pose"]).save(paths["garment_pose"], optimize=True)
    Image.fromarray(arrays["parse"]).save(paths["parse"], optimize=True)


def run(
    cfg: Config,
    csv_path: Path,
    cache_dir: Path,
    device: str,
    logger,
    limit: Optional[int] = None,
    force: bool = False,
) -> Dict:
    df = pd.read_csv(csv_path, dtype={"id": str})
    if limit:
        df = df.head(limit)
    for s in SUBDIRS:
        (cache_dir / s).mkdir(parents=True, exist_ok=True)

    todo = df if force else df[[not is_cached(cache_dir, i) for i in df.id]]
    logger.info(
        f"{csv_path.name}: {len(df)} samples, {len(todo)} to process ({len(df) - len(todo)} already cached)"
    )
    if len(todo) == 0:
        return {"processed": 0, "failed": 0, "total": len(df)}

    pre = Preprocessor(cfg, device=device)
    root = Path(cfg.data.dataset_root)
    failed, t0 = [], time.time()

    for i, r in enumerate(todo.itertuples(), 1):
        try:
            arrays = pre.process(root / r.person_path, root / r.cloth_path, garment_is_flat=True)
            write_sample(cache_dir, r.id, arrays)
        except Exception as e:
            failed.append({"id": r.id, "error": f"{type(e).__name__}: {e}"})
            logger.warning(f"preprocess failed for {r.id}: {type(e).__name__}: {e}")
        if i % 100 == 0:
            el = time.time() - t0
            logger.info(
                f"  {i}/{len(todo)}  {el / i:.2f}s/sample  eta {(len(todo) - i) * el / i / 60:.1f} min"
            )

    return {"processed": len(todo) - len(failed), "failed": len(failed), "total": len(df), "failures": failed}


def main():
    ap = argparse.ArgumentParser(description="Build the offline preprocessing cache")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--device", default="cpu", help="cpu keeps VRAM free; cuda is faster")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    cfg.apply_overrides(args.overrides)

    from training.logging_utils import setup_run_logger

    logger = setup_run_logger("preprocess", Path(cfg.log.dir))
    cache_dir = Path(cfg.data.cache_dir) / f"{cfg.data.height}x{cfg.data.width}"
    logger.info(f"cache dir: {cache_dir}   device: {args.device}")

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", Path(cfg.data.train_csv)))
    if args.split in ("val", "both"):
        splits.append(("val", Path(cfg.data.val_csv)))

    summary = {}
    for name, path in splits:
        if not path.exists():
            logger.error(f"{path} not found — run `python -m datasets.clean` first")
            continue
        summary[name] = run(cfg, path, cache_dir, args.device, logger, args.limit, args.force)
        logger.info(f"{name}: processed {summary[name]['processed']}, failed {summary[name]['failed']}")

    (cache_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"summary -> {cache_dir / 'preprocess_summary.json'}")


if __name__ == "__main__":
    main()
