"""
PyTorch Dataset / DataLoader for saree try-on fine-tuning.

Reads the preprocessing cache (see preprocess.py) and emits exactly the tensors
`TryOnModel.forward` consumes, normalised the same way `TryOnPipeline` normalises them
so training and inference see identical distributions:

    person        (3, H, W)  float, [-1, 1]   TARGET (x1) — the person wearing the saree
    ca_image      (3, H, W)  float, [-1, 1]   clothing-agnostic person (conditioning)
    garment       (3, H, W)  float, [-1, 1]   flat saree
    person_pose   (1, H, W)  float, [-1, 1]   DWPose skeleton
    garment_pose  (1, H, W)  float, [-1, 1]   DWPose skeleton or dummy
    category      ()         int64            1/2/3 (one-pieces = 3)

The clothing-agnostic image is built at load time from the cached parse map rather than
being cached itself, so mask parameters stay tunable without re-running the ~hours-long
preprocessing pass.

Corrupted or missing samples are skipped rather than crashing a long run: a sample that
fails to load is replaced by another index and the failure is counted.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE

from ...preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    create_clothing_agnostic_image,
)
from ..config import Config
from .preprocess import cache_paths, is_cached

CATEGORY_TO_LABEL = {"tops": 1, "bottoms": 2, "one-pieces": 3}


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """HWC or HW uint8 -> CHW float in [-1, 1] (matches utils.tensor.normalize_uint8_to_neg1_1)."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    else:
        t = t.unsqueeze(0)
    return t.float().div_(127.5).sub_(1.0)


class SareeVTONDataset(Dataset):
    """Paired saree try-on dataset backed by the preprocessing cache."""

    def __init__(self, cfg: Config, csv_path: str | Path, split: str = "train",
                 logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.split = split
        self.logger = logger or logging.getLogger(f"dataset.{split}")
        self.cache_dir = Path(cfg.data.cache_dir) / f"{cfg.data.height}x{cfg.data.width}"

        df = pd.read_csv(csv_path, dtype={"id": str})
        limit = cfg.data.max_train_samples if split == "train" else cfg.data.max_val_samples
        if limit:
            df = df.head(limit)

        # Only keep rows whose cache is complete — a partially written cache would
        # otherwise surface as random mid-epoch failures.
        present = [is_cached(self.cache_dir, i) for i in df.id]
        n_missing = len(df) - int(np.sum(present))
        if n_missing:
            self.logger.warning(
                f"{split}: {n_missing}/{len(df)} samples missing from cache {self.cache_dir} "
                f"— excluded. Run the preprocess step to include them."
            )
        self.df = df[present].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(
                f"No usable samples for split={split!r}. Cache dir {self.cache_dir} is empty "
                f"or does not match {cfg.data.height}x{cfg.data.width}."
            )

        self.category = cfg.data.category
        self.category_label = CATEGORY_TO_LABEL[self.category]
        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(self.category)
        self.body_coverage = body_coverage
        labels = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage)
        self.label_ids = [FASHN_LABELS_TO_IDS[x] for x in labels]

        self._fail_lock = threading.Lock()
        self.n_failures = 0
        self.logger.info(f"{split}: {len(self.df)} samples  ({cfg.data.height}x{cfg.data.width}, "
                         f"category={self.category}, agnostic_mask={cfg.data.use_agnostic_mask})")

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------ loading
    def _load(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sid = row.id
        p = cache_paths(self.cache_dir, sid)

        person = np.array(Image.open(p["person"]).convert("RGB"))
        garment = np.array(Image.open(p["garment"]).convert("RGB"))
        p_pose = np.array(Image.open(p["person_pose"]).convert("L"))
        g_pose = np.array(Image.open(p["garment_pose"]).convert("L"))
        parse = np.array(Image.open(p["parse"]).convert("L"))

        H, W = self.cfg.data.height, self.cfg.data.width
        if person.shape[:2] != (H, W):
            raise ValueError(f"{sid}: cached person is {person.shape[:2]}, expected {(H, W)}")
        if parse.shape[:2] != (H, W):
            raise ValueError(f"{sid}: cached parse is {parse.shape[:2]}, expected {(H, W)}")

        # Clothing-agnostic conditioning. With use_agnostic_mask=False the model sees the
        # person still wearing the target saree and can copy it (CONTEXT.md §13.3), which
        # is not a learning signal — hence masking is the training default.
        ca = create_clothing_agnostic_image(
            img_np=person.copy(),
            seg_pred=parse.copy(),
            labels_to_segment_indices=list(self.label_ids),
            body_coverage=self.body_coverage,
            disable_masking=not self.cfg.data.use_agnostic_mask,
            logger=self.logger,
        )

        if self.cfg.data.hflip_prob > 0 and self.split == "train":
            if np.random.rand() < self.cfg.data.hflip_prob:
                person, garment, ca = person[:, ::-1], garment[:, ::-1], ca[:, ::-1]
                p_pose, g_pose = p_pose[:, ::-1], g_pose[:, ::-1]

        return {
            "person": _to_tensor(person),
            "garment": _to_tensor(garment),
            "ca_image": _to_tensor(ca),
            "person_pose": _to_tensor(p_pose),
            "garment_pose": _to_tensor(g_pose),
            "category": torch.tensor(self.category_label, dtype=torch.long),
            "id": sid,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a sample, falling back to neighbours on failure.

        A single unreadable JPEG should not kill a multi-hour run. We try a bounded
        number of deterministic alternates so behaviour stays reproducible, then give up
        loudly rather than returning garbage.
        """
        n = len(self.df)
        for attempt in range(5):
            j = (idx + attempt * 7919) % n  # coprime stride -> deterministic, well spread
            try:
                return self._load(j)
            except Exception as e:
                with self._fail_lock:
                    self.n_failures += 1
                self.logger.warning(
                    f"sample {self.df.iloc[j].id} failed ({type(e).__name__}: {e}); "
                    f"substituting (attempt {attempt+1}/5)"
                )
        raise RuntimeError(f"5 consecutive sample load failures starting at index {idx}; "
                           f"the cache is likely corrupt.")


def collate(batch: List[Dict]) -> Dict:
    """Stack tensors; keep ids as a plain list."""
    out: Dict = {}
    for k in batch[0]:
        if k == "id":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def seed_worker(worker_id: int) -> None:
    """Give each worker a distinct but run-reproducible numpy/random seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    import random

    random.seed(seed)
    cv2.setNumThreads(0)  # avoid oversubscription against the DataLoader workers


def build_dataloader(cfg: Config, split: str = "train", logger=None,
                     shuffle: Optional[bool] = None) -> DataLoader:
    csv_path = cfg.data.train_csv if split == "train" else cfg.data.val_csv
    ds = SareeVTONDataset(cfg, csv_path, split=split, logger=logger)

    is_train = split == "train"
    nw = cfg.data.num_workers
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)

    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=nw,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.data.persistent_workers and nw > 0,
        prefetch_factor=cfg.data.prefetch_factor if nw > 0 else None,
        drop_last=cfg.data.drop_last and is_train,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=g,
    )
