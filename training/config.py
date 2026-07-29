"""
Configuration system for FASHN-VTON saree fine-tuning.

Nothing is hardcoded in the training code: every knob lives here, is serialisable to
YAML, and is round-tripped into every checkpoint so a run is fully reproducible from
its checkpoint alone.

Load order (later wins):
    dataclass defaults  ->  YAML file  ->  CLI --set key=value overrides
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# --------------------------------------------------------------------------------------
# LoRA target presets.
#
# Parameter shares measured on the shipped checkpoint (971.8 M total):
#   qkv+proj             10.79 %   attention projections (double blocks)
#   linear1+linear2      40.46 %   fused attn/MLP (single blocks + patch mixer)
#   img/txt_mlp          21.56 %   double-block MLPs
#   lin (modulation)     26.29 %   adaLN modulation, driven by the class embedding
# --------------------------------------------------------------------------------------
LORA_PRESETS: Dict[str, List[str]] = {
    # attention projections only — cheapest, most conservative
    "attention": ["qkv", "proj", "linear1", "linear2"],
    # + the double-block MLPs. Default: good capacity/cost balance.
    "attention_mlp": [
        "qkv",
        "proj",
        "linear1",
        "linear2",
        "img_mlp.0",
        "img_mlp.2",
        "txt_mlp.0",
        "txt_mlp.2",
    ],
    # + adaLN modulation. `lin` is fed by the timestep+class vector, so this is the
    # path most able to change *category* behaviour — relevant because the saree task
    # is fighting the `one-pieces` dress prior (CONTEXT.md §16.7).
    "attention_mlp_mod": [
        "qkv",
        "proj",
        "linear1",
        "linear2",
        "img_mlp.0",
        "img_mlp.2",
        "txt_mlp.0",
        "txt_mlp.2",
        "lin",
    ],
}


@dataclass
class DataConfig:
    dataset_root: str = "dataset"
    index: str = "dataset/index.parquet"
    train_csv: str = "data_clean/clean_train.csv"
    val_csv: str = "data_clean/clean_validation.csv"
    cache_dir: str = "data_cache"

    # Model input geometry. Must both be divisible by patch_size (12).
    height: int = 864
    width: int = 576

    # "one-pieces" is the nearest supported class for a saree (CONTEXT.md §8.1/§16.7).
    category: str = "one-pieces"

    # If False, the person image is fed unmasked (repo default `segmentation_free=True`)
    # and the model can trivially copy the answer. Training with masking ON is the
    # honest objective — see CONTEXT.md §13.3.
    use_agnostic_mask: bool = True

    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True

    # Augmentation (kept minimal; VTON is geometry-sensitive)
    hflip_prob: float = 0.0

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


@dataclass
class CleaningConfig:
    """
    Thresholds for the garment-input cleaning cascade.

    Only the filters below marked VALIDATED are enabled by default. The sketch and
    collage detectors are present but DISABLED: measured against the dataset they do
    not work (CONTEXT.md §21.2). Their statistics are still computed and stored so a
    future labelled classifier has features to train on.
    """

    # VALIDATED. Rejects cloth images that are actually on-model shots. Phase 2 measured
    # 19.2% prevalence; reproduced at ~27% of the post-garment_score pool.
    # Share of pixels the parser labels as any body part (face/hair/arms/hands/legs/
    # feet/torso). Broader than face+hair alone, which misses mannequins.
    max_person_fraction: float = 0.002

    # VALIDATED (Phase 4). Rejects blouse-piece product renders being supplied as the
    # garment image — ~34% of clean_train by visual count. Signature: parser sees only a
    # `top` covering 10-35% of the frame. See CONTEXT.md §26.3.
    reject_blouse_renders: bool = True
    blouse_top_lo: float = 0.10
    blouse_top_hi: float = 0.35

    # VALIDATED — dataset index flags.
    require_hi_res: bool = True
    drop_duplicates: bool = True
    min_garment_score: float = 0.0  # >0 == "genuine full garment" per DATASET.md

    # NOT VALIDATED — off by default. `flat_fraction > 0.70` matches 26% of the dataset
    # (mostly legitimate flat-lays on plain backdrops) while the one known sketch sits
    # at only ~p85. Enabling this deletes a quarter of the data for no benefit.
    enable_sketch_filter: bool = False
    max_flat_fraction: float = 0.90
    min_colorfulness: float = 8.0

    # NOT VALIDATED — off by default. Otsu merges the known collage into a single
    # component, so the count never fires; no simple global statistic separated it.
    enable_collage_filter: bool = False
    max_foreground_components: int = 3
    min_component_area_frac: float = 0.02

    val_fraction: float = 0.05
    seed: int = 1337


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    preset: str = "attention_mlp"
    # Explicit list overrides `preset` when non-empty.
    target_modules: List[str] = field(default_factory=list)
    # Restrict to these top-level blocks; empty == all.
    include_blocks: List[str] = field(
        default_factory=lambda: ["double_blocks", "single_blocks", "x_patch_mixer"]
    )
    init_scale: float = 0.01  # std for the A matrix; B starts at zero so the model is unchanged at step 0

    def resolve_targets(self) -> List[str]:
        if self.target_modules:
            return list(self.target_modules)
        if self.preset not in LORA_PRESETS:
            raise ValueError(f"Unknown lora preset {self.preset!r}; choose from {sorted(LORA_PRESETS)}")
        return list(LORA_PRESETS[self.preset])


@dataclass
class OptimConfig:
    optimizer: str = "adamw8bit"  # adamw8bit | adamw
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    scheduler: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 100
    min_lr_ratio: float = 0.05


@dataclass
class TrainConfig:
    epochs: int = 10
    max_steps: Optional[int] = None
    batch_size: int = 1  # hard VRAM limit on a 4 GB card
    grad_accum_steps: int = 8  # effective batch 8

    # Rectified-flow objective (CONTEXT.md §6.2)
    timestep_sampling: str = "shifted"  # shifted | uniform | logit_normal
    time_shift_mu: float = 1.5  # matches the inference schedule
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # CFG dropout — the `mask` hook already exists in TryOnModel.forward
    cond_dropout_prob: float = 0.1

    loss: str = "mse"  # mse | huber
    huber_delta: float = 1.0
    # Optional per-timestep loss weighting
    loss_weighting: str = "uniform"  # uniform | snr

    seed: int = 42


@dataclass
class MemoryConfig:
    precision: str = "bf16"  # bf16 | fp16 | fp32 ("auto" resolves at runtime)
    gradient_checkpointing: bool = True
    # Free the CUDA cache every N optimiser steps. 0 disables.
    empty_cache_every: int = 50
    gc_every: int = 200
    # Retry a step at reduced cost after an OOM instead of crashing the run.
    oom_retry: bool = True
    oom_max_retries: int = 2
    # expandable_segments cuts fragmentation materially on small cards.
    set_alloc_conf: bool = True
    # Keep auxiliary preprocessing models off the GPU (Phase 1 §9.5).
    aux_device: str = "cpu"


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    save_every_steps: int = 500
    save_every_epochs: int = 1
    keep_last: int = 3
    save_best: bool = True
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    resume: str = "auto"  # auto | none | <path to checkpoint>


@dataclass
class LogConfig:
    dir: str = "logs"
    run_name: Optional[str] = None
    log_every_steps: int = 10
    val_every_steps: int = 0  # 0 == validate only at epoch end
    tensorboard: bool = True
    log_system_metrics: bool = True
    sample_every_steps: int = 0  # 0 disables periodic image sampling


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: ["ssim", "psnr", "lpips"])
    num_timesteps: int = 30
    guidance_scale: float = 1.5
    seed: int = 42
    max_samples: Optional[int] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: LogConfig = field(default_factory=LogConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    weights_dir: str = "weights"

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return _build(cls, d or {})

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def apply_overrides(self, overrides: List[str]) -> "Config":
        """Apply `--set a.b=value` style overrides in place, with type coercion."""
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Override {ov!r} must be key=value")
            key, raw = ov.split("=", 1)
            target: Any = self
            parts = key.split(".")
            for p in parts[:-1]:
                if not hasattr(target, p):
                    raise ValueError(f"Unknown config section {p!r} in {key!r}")
                target = getattr(target, p)
            leaf = parts[-1]
            if not hasattr(target, leaf):
                raise ValueError(f"Unknown config key {key!r}")
            setattr(target, leaf, _coerce(getattr(target, leaf), raw))
        self.validate()
        return self

    # ---------------------------------------------------------------- validation
    def validate(self) -> "Config":
        patch = 12
        if self.data.height % patch or self.data.width % patch:
            raise ValueError(
                f"height/width must be divisible by patch_size={patch}; "
                f"got {self.data.height}x{self.data.width}"
            )
        if self.train.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.train.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.lora.rank < 1:
            raise ValueError("lora.rank must be >= 1")
        if self.memory.precision not in ("bf16", "fp16", "fp32", "auto"):
            raise ValueError(f"unknown precision {self.memory.precision!r}")
        if self.train.timestep_sampling not in ("shifted", "uniform", "logit_normal"):
            raise ValueError(f"unknown timestep_sampling {self.train.timestep_sampling!r}")
        if not 0.0 <= self.train.cond_dropout_prob <= 1.0:
            raise ValueError("cond_dropout_prob must be in [0, 1]")
        self.lora.resolve_targets()  # raises on a bad preset
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.train.batch_size * self.train.grad_accum_steps


# -------------------------------------------------------------------------- helpers
def _build(cls, d: Dict[str, Any]):
    """
    Recursively construct `cls` from a plain dict.

    Field *types* are unreliable here: `from __future__ import annotations` turns them
    into strings, so `is_dataclass(f.type)` is always False. We therefore build the
    object with its defaults first and inspect the resulting *instances*, which are
    real dataclass objects, to decide where to recurse.
    """
    obj = cls()
    known = {f.name for f in fields(cls)}
    for key, val in (d or {}).items():
        if key not in known:
            raise ValueError(
                f"unknown config key {key!r} for {cls.__name__}; expected one of {sorted(known)}"
            )
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            setattr(obj, key, _build(type(cur), val))
        else:
            setattr(obj, key, val)
    return obj


def _coerce(current: Any, raw: str) -> Any:
    """Coerce a CLI string to the type of the value it replaces."""
    # String fields take the literal text: `checkpoint.resume=none` must stay the
    # string "none" (a valid sentinel), not become None.
    if isinstance(current, str):
        return raw
    if raw.lower() in ("none", "null"):
        return None
    if current is None:
        if raw == "":
            return None
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                pass
        return raw
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        if raw.strip() in ("", "[]"):
            return []
        items = [x.strip() for x in raw.strip("[]").split(",") if x.strip()]
        # Match the element type of the existing list, otherwise numeric lists like
        # optim.betas come back as strings and blow up inside the optimiser.
        if current and isinstance(current[0], bool):
            return [x.lower() in ("1", "true", "yes", "y", "on") for x in items]
        if current and isinstance(current[0], (int, float)) and not isinstance(current[0], bool):
            caster = float if isinstance(current[0], float) else int
            return [caster(x) for x in items]
        return items
    return raw
