"""
Training package for FASHN-VTON saree fine-tuning.

The upstream repository is inference-only; everything here was written for this
project (see CONTEXT.md §6 for how the objective was derived from the inference
contract, and §19-24 for the Phase 3 design).

Layout
------
    config.py          typed, YAML-serialisable configuration
    data/clean.py      garment-input cleaning cascade -> clean_train/validation.csv
    data/preprocess.py offline pose/parse/image cache
    data/dataset.py    Dataset + DataLoader
    lora.py            adapter injection, save/load, merge-and-export
    losses.py          rectified-flow velocity objective + CFG dropout
    memory.py          precision, checkpointing, OOM handling, VRAM accounting
    checkpoint.py      atomic checkpointing + full resume
    logging_utils.py   console/file/TensorBoard logging + system telemetry
    engine.py          model assembly, optimiser/scheduler, train/val loops
    train.py           CLI entrypoint
"""

from .config import LORA_PRESETS, Config

__all__ = ["Config", "LORA_PRESETS"]
