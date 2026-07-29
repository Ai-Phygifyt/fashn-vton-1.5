# Training notes

## Objective

The upstream model publishes no training code, so the objective was reconstructed from the
inference implementation. It is rectified flow with a velocity target:

```
x0 ~ N(0, I)                      noise
t  ~ p(t)                         timestep in (0, 1)
xt = (1 - t)·x0 + t·x1            linear interpolant
v* = x1 - x0                      target
L  = || model(xt, t, cond) - v* ||²
```

`t = 0` is noise, `t = 1` is data. Verified empirically: scoring the untouched pretrained
model under three competing formulations gave **0.0795** for this objective against **1.10**
(data prediction) and **4.36** (noise prediction) — a pretrained model can only score near
zero under the objective it was actually trained with.

Timestep sampling defaults to `shifted`, reusing the inference schedule's own
reparameterisation so training and sampling agree about where model capacity is spent.

## Memory

| Technique | Effect |
|---|---|
| LoRA | 29.8 M trainable instead of 972 M; full fine-tuning would need ~17.5 GB |
| Gradient checkpointing | ~35 % extra compute for a large activation saving — **mandatory below 8 GB** |
| 8-bit AdamW | Halves optimiser state; checkpoints 358 → 180 MB |
| Gradient accumulation | Effective batch of 4 at the memory cost of 1 |
| bf16 | Native checkpoint dtype; no loss scaler needed |
| Offline preprocessing | Pose and segmentation computed once, not per epoch |

Measured peak VRAM (RTX 3050, batch 1, LoRA rank 32):

| Resolution | Peak VRAM | s / step |
|---|---:|---:|
| 864×576 | 3.39 GB | 24.0 |
| 792×528 | 3.19 GB | 5.6 |
| 648×432 | 3.00 GB | 3.4 |
| 576×384 | 2.67 GB | 2.5 |
| 864×576, checkpointing off | OOM | — |

The 864×576 row is the important one: it fits, but runs sevenfold slower than 648×432 for
less than twice the computation. At ~92 % memory occupancy the allocator thrashes. On GPUs
with more headroom this cliff disappears.

## Hyperparameters

Validated by controlled experiment — a single variable changed between runs.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Learning rate | 5 × 10⁻⁴ | **1 × 10⁻⁴** |
| Steps completed | 234 (terminated) | 450 |
| Best validation loss | 0.0321 | **0.0292** |
| Behaviour | diverged from ~step 150 | stable, monotonic |

`optim.lr = 1e-4` is the validated setting. Weight decay is 0 and CFG dropout is 0 in
`phase4_overfit.yaml` because both work against memorisation, which is what a validation run
is testing. **For full training, restore `train.cond_dropout_prob = 0.1`** — without it the
unconditional branch is never trained and classifier-free guidance degrades at inference.

## Checkpoints

Contain adapter weights, optimiser state, scheduler state, epoch and step counters, the full
configuration, and RNG state — about 180 MB. The 1.94 GB base model is not stored; it is
reloaded from `weights/`, which also guarantees a resumed run starts from an identical base.

Writes are atomic (temp file + rename), so a crash during saving cannot corrupt the recovery
point. Verified: resume after graceful interruption and after forced process termination both
restore the run exactly.

## Scaling to a larger GPU

1. Raise `train.batch_size` above 1 and lower `train.grad_accum_steps` proportionally — the
   single largest throughput gain available.
2. Train at native 864×576 once you have headroom.
3. Consider disabling gradient checkpointing above ~16 GB for a ~35 % speedup.
4. Restore `train.cond_dropout_prob = 0.1` and a non-zero `optim.weight_decay` for
   generalisation rather than memorisation.
5. Use `optim.scheduler = cosine` with warmup instead of the constant schedule used for
   validation runs.

## Evaluation caveat

Similarity metrics are weak indicators of garment correctness here. In baseline evaluation
the single worst failure — a saree rendered as a tunic and trousers — recorded the **highest**
SSIM of all 240 generations, because pixel metrics are dominated by background, skin and hair
which the model reproduces faithfully regardless.

Use `masked_ssim` / `masked_lpips` (which score only the regenerated region) alongside visual
review. `inference/metrics.py` is a registry with registered stubs for `saree_structure` and
`pallu_presence` — implementing those is the recommended next step.
