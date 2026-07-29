#!/usr/bin/env python3
"""
Inference entrypoint — generate a try-on image from a person and a garment photograph.

    # base pretrained model
    python inference.py --person person.jpg --garment saree.jpg --out result.png

    # fine-tuned LoRA checkpoint
    python inference.py --person person.jpg --garment saree.jpg \
        --lora checkpoints/<run>/best.pt --out result.png

    # batch over a directory of garments
    python inference.py --person person.jpg --garment-dir sarees/ --out-dir results/
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

from PIL import Image


def build_pipeline(
    weights_dir: str, height: int, width: int, aux_device: str = "cpu", device: str | None = None
):
    """
    Construct the try-on pipeline with the low-VRAM adjustments.

    Two changes relative to a naive construction, both required to fit a 4 GB card:
    the auxiliary pose and parsing models are placed on the CPU (they run once per
    image, before sampling), and the base state dict is staged through CPU rather than
    being loaded straight to the GPU, which would transiently need ~3.9 GB.
    """
    import torch  # noqa: F401  (imported for side effects / availability check)

    from inference.pipeline import TryOnPipeline

    class _Pipeline(TryOnPipeline):
        def __init__(self, *a, aux_device: str = "cpu", shape=(864, 576), **kw):
            self._aux, self._shape = aux_device, shape
            super().__init__(*a, **kw)

        def _setup_tryon_model(self):
            from models.tryon_mmdit import TryOnModel
            from utils import load_checkpoint

            self.tryon_model = TryOnModel(input_shape=self._shape)
            sd = load_checkpoint(os.path.join(self.weights_dir, "model.safetensors"), device="cpu")
            self.tryon_model.load_state_dict(sd)
            del sd
            gc.collect()
            self.tryon_model.to(self.device, dtype=self.inference_dtype).eval()

        def _setup_pose_model(self):
            from models.dwpose import DWposeDetector

            d = "cuda:0" if self._aux == "cuda" else self._aux
            self.pose_model = DWposeDetector(
                checkpoints_dir=os.path.join(self.weights_dir, "dwpose"), device=d
            )

        def _setup_hp_model(self):
            from fashn_human_parser import FashnHumanParser

            self.hp_model = FashnHumanParser(device=self._aux)

    return _Pipeline(weights_dir=weights_dir, device=device, aux_device=aux_device, shape=(height, width))


def apply_lora(pipe, checkpoint: str, verbose: bool = True) -> dict:
    """Load LoRA adapters from a checkpoint and merge them into the base weights."""
    import torch

    from lora import inject_lora, load_lora_state_dict, merge_lora_into_base
    from training.config import Config

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    inject_lora(pipe.tryon_model, cfg.lora)
    load_lora_state_dict(pipe.tryon_model, ck["lora"], strict=True)
    n = merge_lora_into_base(pipe.tryon_model)
    pipe.tryon_model.to(pipe.device, dtype=pipe.inference_dtype).eval()
    if verbose:
        print(f"applied {n} LoRA modules from {checkpoint} (epoch {ck.get('epoch')}, step {ck.get('step')})")
    return ck


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Saree virtual try-on inference")
    ap.add_argument("--person", required=True, help="path to the person photograph")
    ap.add_argument("--garment", help="path to the garment photograph")
    ap.add_argument("--garment-dir", help="directory of garments to run in batch")
    ap.add_argument("--lora", default=None, help="LoRA checkpoint; omit for the base model")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--out-dir", default=None, help="output directory for --garment-dir")
    ap.add_argument("--category", default="one-pieces", choices=["one-pieces", "tops", "bottoms"])
    ap.add_argument(
        "--garment-photo-type",
        default="flat-lay",
        choices=["flat-lay", "model"],
        help="'flat-lay' for a product photo, 'model' if worn by someone",
    )
    ap.add_argument("--height", type=int, default=648)
    ap.add_argument("--width", type=int, default=432)
    ap.add_argument("--steps", type=int, default=30, help="20 fast, 30 balanced, 50 quality")
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-mask", action="store_true", help="skip masking the existing garment (the model may copy it)"
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--aux-device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args(argv)

    if not args.garment and not args.garment_dir:
        ap.error("provide --garment or --garment-dir")
    if not Path(args.weights_dir, "model.safetensors").exists():
        sys.exit(
            f"model weights not found in {args.weights_dir}\n"
            f"run: python scripts/download_weights.py --weights-dir {args.weights_dir}"
        )

    person = Image.open(args.person).convert("RGB")
    garments = (
        [Path(args.garment)]
        if args.garment
        else sorted(
            p
            for p in Path(args.garment_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    )

    print(f"loading pipeline ({args.height}x{args.width}, auxiliary models on {args.aux_device})")
    t0 = time.time()
    pipe = build_pipeline(args.weights_dir, args.height, args.width, args.aux_device, args.device)
    if args.lora:
        apply_lora(pipe, args.lora)
    print(f"ready in {time.time() - t0:.1f}s")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for g in garments:
        t = time.time()
        result = pipe(
            person_image=person,
            garment_image=Image.open(g).convert("RGB"),
            category=args.category,
            garment_photo_type=args.garment_photo_type,
            num_samples=1,
            num_timesteps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            segmentation_free=args.no_mask,
        )
        dst = out_dir / f"{g.stem}.png" if out_dir else Path(args.out)
        result.images[0].save(dst)
        print(f"  {g.name} -> {dst}  ({time.time() - t:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
