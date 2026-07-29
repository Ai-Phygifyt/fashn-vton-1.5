#!/usr/bin/env python3
"""
Phase 2 baseline inference over the stratified saree evaluation subset.

The repo's TryOnPipeline is used AS-IS — model behaviour is not modified. The only
override is *device placement* for the two auxiliary models (DWPose, human parser),
which Phase 1 measured as the binding VRAM constraint on a 4 GB card. They run once
per image, before sampling, so moving them to CPU costs a little wall-clock and
frees ~0.6 GB plus the ONNX Runtime arena. No weights, shapes, or math change.

Dataset mapping:
    person_path -> person_image   (person wearing THAT saree = also the ground truth)
    cloth_path  -> garment_image  (flat saree)
    category    -> "one-pieces"   (nearest supported class; see CONTEXT.md §8.1)
    photo type  -> "flat-lay"     (cloth images are flat garment shots, not worn)

Because each pair's person image already shows that exact saree, this is the
standard VITON-HD *paired* protocol: the model should reconstruct the person image.
That gives us a real ground truth for SSIM/PSNR/LPIPS — but see --mode:

    segfree   repo default (segmentation_free=True). The model sees the person
              STILL WEARING the target saree, so it can score well by copying.
              Metrics here are an upper bound, not a measure of try-on skill.
    masked    segmentation_free=False. The garment region is masked out first, so
              the model must actually regenerate the saree. This is the honest
              try-on measurement.

Usage:
    python eval/run_baseline.py --mode both --num-timesteps 30
"""

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from fashn_vton import TryOnPipeline


class EvalPipeline(TryOnPipeline):
    """TryOnPipeline with auxiliary models pinned to a chosen device (default CPU).

    Also fixes a load-time VRAM spike in the shipped `_setup_tryon_model`: it calls
    `load_checkpoint(..., device=cuda)`, which puts a full 1.94 GB bf16 state dict on
    the GPU, and *then* moves a separate CPU copy of the model to the GPU — needing
    ~3.9 GB transiently. Staging the state dict through CPU halves that peak. Numerics
    are identical; only the transfer order changes.
    """

    def __init__(self, *a, aux_device: str = "cpu", **kw):
        self._aux_device = aux_device
        super().__init__(*a, **kw)

    def _setup_tryon_model(self):
        import gc
        import os

        from fashn_vton.tryon_mmdit import TryOnModel
        from fashn_vton.utils import load_checkpoint

        model_path = os.path.join(self.weights_dir, "model.safetensors")
        self.logger.info(f"Loading TryOnModel from {model_path} (staged via CPU)")
        self.tryon_model = TryOnModel()
        state_dict = load_checkpoint(model_path, device="cpu")   # <- CPU, not CUDA
        self.tryon_model.load_state_dict(state_dict)
        del state_dict
        gc.collect()
        self.tryon_model.to(self.device, dtype=self.inference_dtype).eval()
        self.logger.info("TryOnModel loaded")

    def _setup_pose_model(self):
        from fashn_vton.dwpose import DWposeDetector
        import os
        d = self._aux_device
        self.logger.info(f"Loading DWPose on {d}")
        self.pose_model = DWposeDetector(
            checkpoints_dir=os.path.join(self.weights_dir, "dwpose"), device=d
        )

    def _setup_hp_model(self):
        from fashn_human_parser import FashnHumanParser
        self.logger.info(f"Loading FashnHumanParser on {self._aux_device}")
        self.hp_model = FashnHumanParser(device=self._aux_device)


MODES = {
    "segfree": dict(segmentation_free=True),
    "masked": dict(segmentation_free=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="eval/eval_subset.csv")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--weights-dir", default="./weights")
    ap.add_argument("--out-dir", default="outputs/phase2")
    ap.add_argument("--mode", default="both", choices=["segfree", "masked", "both"])
    ap.add_argument("--category", default="one-pieces")
    ap.add_argument("--garment-photo-type", default="flat-lay", choices=["model", "flat-lay"])
    ap.add_argument("--cloth-type", default="eval/cloth_type.csv",
                    help="probe output; enables per-sample garment_photo_type")
    ap.add_argument("--auto-photo-type", action="store_true",
                    help="use the probe to pick 'model' vs 'flat-lay' per sample "
                         "(19%% of cloth images are actually on-model shots)")
    ap.add_argument("--num-timesteps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aux-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_dir)
    sub = pd.read_csv(args.subset, dtype={"id": str})
    if args.limit:
        sub = sub.head(args.limit)

    on_model = {}
    if args.auto_photo_type:
        ct = pd.read_csv(args.cloth_type, dtype={"id": str})
        on_model = dict(zip(ct.id, ct.cloth_is_on_model))
        print(f"auto photo-type: {sum(on_model.values())}/{len(on_model)} cloth images are on-model")

    modes = ["segfree", "masked"] if args.mode == "both" else [args.mode]

    print(f"Loading pipeline (aux on {args.aux_device})...")
    t0 = time.time()
    pipe = EvalPipeline(weights_dir=args.weights_dir, aux_device=args.aux_device)
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"Pipeline loaded in {load_s:.1f}s   VRAM after load: {vram_after_load:.2f} GB\n")

    # Preserve prior records when resuming, otherwise a restart wipes the runlog.
    records = []
    runlog_path = out_root / "runlog.csv"
    if args.skip_existing and runlog_path.exists():
        prior = pd.read_csv(runlog_path, dtype={"id": str})
        records = prior.to_dict("records")
        print(f"resuming: carried {len(records)} prior records from {runlog_path}")

    done = {(r["mode"], r["id"]) for r in records}
    for mode in modes:
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(sub.itertuples(), 1):
            sid = row.id
            d = mode_dir / sid
            if args.skip_existing and (d / "output.png").exists() and (mode, sid) in done:
                continue
            d.mkdir(parents=True, exist_ok=True)

            rec = dict(id=sid, mode=mode, source=row.source, is_clean=bool(row.is_clean),
                       fabric_family=row.fabric_family, res_bucket=row.res_bucket,
                       gscore_q=row.gscore_q, garment_score=row.garment_score,
                       target_type=row.target_type, title=row.title)
            try:
                person = Image.open(root / row.person_path).convert("RGB")
                garment = Image.open(root / row.cloth_path).convert("RGB")

                gpt = ("model" if on_model.get(sid, False) else "flat-lay") \
                    if args.auto_photo_type else args.garment_photo_type
                rec["garment_photo_type"] = gpt

                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                out = pipe(
                    person_image=person,
                    garment_image=garment,
                    category=args.category,
                    garment_photo_type=gpt,
                    num_samples=1,
                    num_timesteps=args.num_timesteps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    **MODES[mode],
                )
                dt = time.time() - t
                peak = torch.cuda.max_memory_allocated() / 1e9

                # copy ORIGINALS (never overwrite the dataset)
                shutil.copy2(root / row.person_path, d / "person.jpg")
                shutil.copy2(root / row.cloth_path, d / "garment.jpg")
                out.images[0].save(d / "output.png")

                rec.update(ok=True, seconds=round(dt, 2), peak_vram_gb=round(peak, 3),
                           out_w=out.images[0].width, out_h=out.images[0].height,
                           person_w=row.person_w, person_h=row.person_h, error="")
                print(f"[{mode} {i}/{len(sub)}] {sid}  {dt:5.1f}s  {peak:.2f}GB")
            except Exception as e:
                rec.update(ok=False, seconds=None, peak_vram_gb=None, error=f"{type(e).__name__}: {e}")
                print(f"[{mode} {i}/{len(sub)}] {sid}  FAILED  {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
            records.append(rec)
            pd.DataFrame(records).to_csv(out_root / "runlog.csv", index=False)

    df = pd.DataFrame(records)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "runlog.csv", index=False)
    meta = dict(
        subset=args.subset, n=len(sub), modes=modes, category=args.category,
        garment_photo_type=args.garment_photo_type, num_timesteps=args.num_timesteps,
        guidance_scale=args.guidance_scale, seed=args.seed, aux_device=args.aux_device,
        pipeline_load_seconds=round(load_s, 1), vram_after_load_gb=round(vram_after_load, 3),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    (out_root / "run_config.json").write_text(json.dumps(meta, indent=2))

    ok = df[df.ok] if len(df) else df
    print(f"\n=== DONE ===  {len(ok)}/{len(df)} succeeded")
    if len(ok):
        print(f"mean {ok.seconds.mean():.1f}s   median {ok.seconds.median():.1f}s   "
              f"total {ok.seconds.sum()/60:.1f} min   peak VRAM {ok.peak_vram_gb.max():.2f} GB")


if __name__ == "__main__":
    main()
