# Saree Virtual Try-On

LoRA fine-tuning pipeline for **FASHN-VTON v1.5**, adapted to Indian sarees. The upstream
model ships inference code only; this repository adds the complete training system.

Given a photograph of a person and a photograph of a saree, the model generates the person
wearing that saree. The pretrained model produces Western garments from flat-lay saree
inputs; fine-tuning corrects this.

| Stage | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Pretrained baseline | 0.7693 | 16.35 dB | 0.2743 |
| **Fine-tuned** (450 steps, 24 samples) | **0.7981** | **22.33 dB** | **0.1885** |

Everything runs inside **3.00 GB of VRAM** — verified on an RTX 3050 Laptop (4 GB).

---

# Setup — follow these steps in order

Total time: about 40 minutes, most of it downloading. Every command is run **from the
repository root**.

## Step 1 — Check your system

| Requirement | Needed | Check with |
|---|---|---|
| Python | 3.10 – 3.12 | `python --version` |
| NVIDIA GPU | 4 GB VRAM minimum | `nvidia-smi` |
| CUDA | 11.8 or newer | `nvidia-smi` (top right) |
| Free disk | ~10 GB | `df -h .` |

Python 3.13+ does not yet have wheels for parts of this stack — use 3.12.

<details>
<summary>No Python 3.12? Install an isolated one (click to expand)</summary>

```bash
# Option A — uv (fastest, no system changes)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv          # downloads Python 3.12 automatically

# Option B — pyenv
pyenv install 3.12.13 && pyenv local 3.12.13
```
</details>

## Step 2 — Clone and create the environment

```bash
git clone <repository-url>
cd <repository-name>

python3.12 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
```

Your prompt should now show `(.venv)`.

## Step 3 — Install dependencies

Install PyTorch **first**, matched to your CUDA version — check yours with `nvidia-smi`:

```bash
# CUDA 12.1 or newer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then everything else:

```bash
pip install -r requirements.txt
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected:  True NVIDIA GeForce RTX ...
```

If this prints `False`, your PyTorch build does not match your CUDA driver — reinstall using
the correct index URL above.

## Step 4 — Download the model weights (~2.4 GB)

```bash
python scripts/download_weights.py --weights-dir ./weights
```

This creates:

```
weights/
├── model.safetensors              1.9 GB   the try-on model
└── dwpose/
    ├── yolox_l.onnx                207 MB  person detection
    └── dw-ll_ucoco_384.onnx        135 MB  pose estimation
```

A fourth model (the human parser, ~244 MB) downloads automatically to the HuggingFace cache
on first use.

> If the download stalls, press Ctrl-C and re-run — it resumes from where it stopped.

## Step 5 — Download the dataset (~2.1 GB)

**Download link:** https://drive.google.com/drive/folders/17xyLbR4_OGSTK1HUBS2NgmtZ7mRHBhZY

Download `saree_dataset_2000.tar.gz` from that folder, then extract it **into the repository
root** so that a `dataset/` directory sits next to `train.py`:

```bash
# from the repository root, with the archive downloaded here
tar -xzf saree_dataset_2000.tar.gz

# confirm the layout
ls dataset/
# expected:  metadata.csv  rejected.csv  summary.json  train  train.csv  val  val.csv
```

The result must look exactly like this:

```
<repository-root>/
├── train.py
├── inference.py
├── weights/                    ← from Step 4
└── dataset/                    ← from Step 5
    ├── train/
    │   ├── image/    1800 .jpg files
    │   ├── cloth/    1800 .jpg files
    │   └── pairs.txt
    ├── val/
    │   ├── image/     200 .jpg files
    │   ├── cloth/     200 .jpg files
    │   └── pairs.txt
    ├── train.csv
    ├── val.csv
    └── metadata.csv
```

Now verify it:

```bash
python scripts/validate_dataset.py --root dataset
```

Expected output:

```
  train    1800 pairs   OK
  val       200 pairs   OK
  total    2000 pairs

All checks passed
```

If this fails, do not continue — the error message names the specific file.

## Step 6 — Build the preprocessing cache (~15 minutes, one time)

Pose detection and human parsing are expensive and deterministic, so they are computed once
and cached instead of being repeated every epoch.

```bash
python -m datasets.preprocess --config configs/default.yaml --device cuda
```

Expected: `train: processed 1800, failed 0` and `val: processed 200, failed 0`.

This writes `data_cache/648x432/` (~1.5 GB). It is resumable — re-running skips samples that
are already cached.

> The cache is **keyed by resolution**. If you later change `data.height` / `data.width`, you
> must re-run this step for the new resolution.

## Step 7 — Start training

```bash
python train.py --config configs/default.yaml
```

You should see, within about 30 seconds:

```
device=cuda dtype=torch.bfloat16 effective_batch=4
LoRA: 140 modules adapted (rank 32, alpha 64)
  trainable 29,818,880 / 1,001,632,688 params  (2.977%)
gradient checkpointing enabled on 3 block stacks
optimizer: bitsandbytes AdamW8bit
===== epoch 1/10 =====
epoch 0 | step 1/4500 | loss 0.0296 | lr 2.50e-05 | 14.2s/it | vram 2.15/3.00GB
```

**That is a working training run.** If you got here, setup is complete.

Watch it live with TensorBoard:

```bash
tensorboard --logdir logs/
```

---

# Everyday commands

## Training

```bash
# start
python train.py --config configs/default.yaml

# resume after an interruption or crash
python train.py --config configs/default.yaml --resume auto

# resume from a specific checkpoint
python train.py --config configs/default.yaml --resume checkpoints/<run>/best.pt

# override any setting without editing the config
python train.py --config configs/default.yaml \
    --set optim.lr=5e-5 lora.rank=32 train.epochs=20 data.height=864 data.width=576
```

Interrupting with Ctrl-C writes an emergency checkpoint before exiting, so no progress is
lost. Resume restores adapter weights, optimiser state, scheduler state, epoch/step counters
and RNG state.

## Inference

```bash
# with a fine-tuned checkpoint
python inference.py --person person.jpg --garment saree.jpg \
    --lora checkpoints/<run>/best.pt --out result.png

# with the base pretrained model, for comparison
python inference.py --person person.jpg --garment saree.jpg --out baseline.png

# batch: one person, a whole directory of sarees
python inference.py --person person.jpg --garment-dir sarees/ \
    --lora checkpoints/<run>/best.pt --out-dir results/
```

Important flags:

| Flag | Meaning |
|---|---|
| `--garment-photo-type flat-lay\|model` | **Must match your garment image.** `flat-lay` for a product photo of the fabric, `model` if someone is wearing it |
| `--no-mask` | Skip masking the existing garment. The model may then copy what the person is already wearing |
| `--height --width` | Generation resolution. Default 648×432 |
| `--steps` | 20 fast · 30 balanced · 50 quality |

## Evaluation

```bash
python -m inference.evaluate --lora checkpoints/<run>/best.pt \
    --subset dataset/val.csv --masked --out outputs/eval_ft

python -m inference.evaluate --compare outputs/eval_base outputs/eval_ft
```

## Tests

```bash
pytest tests/ -q          # expect 91 passed
```

---

# Where things are written

| Path | Contents |
|---|---|
| `checkpoints/<run>/latest.pt` | Most recent checkpoint — used by `--resume auto` |
| `checkpoints/<run>/best.pt` | Best validation loss |
| `checkpoints/<run>/epoch*.pt` | Per-epoch snapshots (rotated; `keep_last` in config) |
| `logs/<run>/train.log` | Full training log |
| `logs/<run>/tb/` | TensorBoard event files |
| `logs/<run>/config.yaml` | Exact configuration used for that run |
| `logs/<run>/summary.json` | Final status, steps, elapsed, peak VRAM, OOM count |
| `data_cache/<HxW>/` | Preprocessing cache |
| `outputs/` | Evaluation results and comparison boards |

Checkpoints hold **adapter weights only** (~180 MB), not the 1.94 GB base model — that is
reloaded from `weights/`, which also guarantees a resumed run starts from an identical base.

---

# Hardware guidance

| GPU | VRAM | Recommended resolution | Time per epoch (2,000 samples) |
|---|---|---|---|
| RTX 3050 Laptop | 4 GB | 648×432 | ~5 h |
| RTX 3060 / 4060 | 8 GB | 648×432 – 864×576 | ~2–3 h |
| RTX 3090 / 4090 | 24 GB | 864×576 | ~40 min |
| A100 | 40–80 GB | 864×576, larger batch | ~15 min |

**A measured warning about resolution.** On a 4 GB card, training at the model's native
864×576 fits in memory but runs at **24.0 s per step versus 3.4 s at 648×432** — a sevenfold
penalty for less than twice the computation. At ~92 % memory occupancy the allocator thrashes
rather than computing. On GPUs with more headroom this cliff disappears and native resolution
is preferable. Measure before committing to a long run.

**On a larger GPU, do these three things first:**

1. Raise `train.batch_size` above 1 and lower `train.grad_accum_steps` proportionally — the
   single largest throughput gain available.
2. Restore `train.cond_dropout_prob=0.1`. It is 0 in the validation config because it works
   against memorisation; without it the unconditional branch is never trained and
   classifier-free guidance degrades at inference.
3. Switch `optim.scheduler` to `cosine` with warmup, and set a non-zero `optim.weight_decay`
   for generalisation rather than memorisation.

---

# Troubleshooting

**`CUDA out of memory`**
Lower the resolution (`--set data.height=576 data.width=384`), confirm
`memory.gradient_checkpointing=true` (mandatory below 8 GB), keep `train.batch_size=1`, and
export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching. The pipeline
retries a step after an OOM before skipping it, and reports both counts in `summary.json`.

**`No usable samples for split='train'`**
The preprocessing cache is missing or was built at a different resolution. The cache is keyed
by resolution (`data_cache/648x432/`), so changing `data.height`/`data.width` requires
re-running Step 6.

**`torch.cuda.is_available()` returns False**
Your PyTorch build does not match your CUDA driver. Reinstall with the correct index URL from
Step 3.

**Training is far slower than expected**
Check whether you are at the memory-pressure cliff described above. Also check GPU
temperature in the log — sustained load throttles laptop GPUs by more than 2×.

**`optimizer: torch AdamW` instead of `AdamW8bit`**
`bitsandbytes` is not installed. Training still works but uses roughly twice the optimiser
memory. `pip install bitsandbytes`.

**`DWposeDetector` fails on CUDA**
`onnxruntime-gpu` needs a matching CUDA/cuDNN. Either fix the CUDA install or run the
auxiliary models on CPU — `memory.aux_device=cpu`, which is the default and costs only a few
seconds per image.

**Resume starts from scratch**
`--resume auto` looks for `checkpoints/<run>/latest.pt`. The run name must match — pass
`--run-name <name>`, or point `--resume` at an explicit checkpoint path.

**`ModuleNotFoundError: No module named 'models'`**
Run commands from the repository root. Packages are resolved by path, not installed.

---

# Repository layout

```
.
├── configs/              training configurations (YAML)
│   ├── default.yaml          full-dataset training
│   └── phase4_overfit.yaml   small-scale validation run
├── datasets/             dataset pipeline
│   ├── clean.py              garment-input cleaning cascade
│   ├── preprocess.py         offline pose / segmentation cache
│   └── dataset.py            Dataset + DataLoader
├── models/               model definitions
│   ├── tryon_mmdit.py        the try-on transformer (972 M params)
│   ├── preprocessing/        masks, transforms, clothing-agnostic construction
│   └── dwpose/               pose detection (vendored)
├── lora/                 LoRA adapters — injection, save/load, merge-and-export
├── training/             training system
│   ├── config.py             typed configuration
│   ├── engine.py             optimiser, scheduler, train / validation loops
│   ├── losses.py             rectified-flow objective + CFG dropout
│   ├── memory.py             precision, checkpointing, OOM handling
│   ├── checkpoint.py         atomic checkpointing + resume
│   └── logging_utils.py      console / file / TensorBoard + GPU telemetry
├── inference/            inference and evaluation
│   ├── pipeline.py           the try-on pipeline
│   ├── metrics.py            pluggable metric registry
│   └── evaluate.py           evaluation driver + comparison boards
├── utils/                checkpoint I/O, sampling schedule, tensor helpers, logging
├── scripts/
│   ├── download_weights.py       fetch pretrained weights
│   ├── build_handoff_dataset.py  rebuild the training set from raw data
│   └── validate_dataset.py       verify a dataset before training
├── tests/                91 automated tests
├── docs/
│   ├── DATASET.md            data provenance, cleaning, known limitations
│   └── TRAINING.md           objective, memory techniques, hyperparameters
├── checkpoints/          created at runtime
├── train.py              training entrypoint
├── inference.py          inference entrypoint
└── requirements.txt
```

---

# Dataset format

```
dataset/
├── train/
│   ├── image/<id>.jpg      person wearing the saree
│   ├── cloth/<id>.jpg      flat saree
│   └── pairs.txt           "<id>.jpg <id>.jpg" per line
├── val/  (same structure)
├── train.csv               index the training pipeline reads
├── val.csv
└── metadata.csv            provenance and quality fields
```

`image/<id>.jpg` and `cloth/<id>.jpg` share the same id — that is the pairing.

**There is no separate `gt/` directory, and none is needed.** In paired virtual try-on,
`image/<id>.jpg` *is* the ground truth: it is a photograph of the person wearing the target
garment. The model input is derived from it at load time by masking out the garment region,
and the unmodified photograph is the reconstruction target. A duplicate `gt/` would double
the dataset for no additional information.

See [docs/DATASET.md](docs/DATASET.md) for how the 2,000 pairs were selected from 14,069
scraped samples, and what was rejected.

---

# Known limitations

- Generated drapes are simplified; distinct waist pleats are not reproduced, and a separate
  fitted blouse appears inconsistently.
- Validation used 450 steps on 24 samples. This establishes that the pipeline learns, not
  that the model is production-ready.
- **Similarity metrics are weak indicators of garment correctness for this task.** In baseline
  evaluation the single worst failure — a saree rendered as a tunic and trousers — recorded
  the *highest* SSIM of all 240 generations, because pixel metrics are dominated by
  background, skin and hair. Use `masked_ssim` / `masked_lpips` alongside visual review, never
  similarity scores alone.
- The model has no saree category and treats a saree as a `one-piece`, which carries a
  stitched-dress prior.
- Rare dataset defects (captioned fabric swatches, technical illustrations, collage images)
  are not detected automatically — roughly 2 % of samples.

---

# Licence

Apache-2.0 — see [LICENSE](LICENSE).

Built on [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) by FASHN AI.
Third-party components: [DWPose](https://github.com/IDEA-Research/DWPose) (Apache-2.0),
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0),
[fashn-human-parser](https://github.com/fashn-AI/fashn-human-parser).

Dataset images are copyrighted product photography collected for research and model training.
Source URLs are preserved per row in `metadata.csv`. Not for public redistribution.
