#!/usr/bin/env python3
"""
Parse a training log into loss curves and a progression board.

Phase 4 Parts 4 & 8: the learning curve is the primary evidence that the pipeline
learns, and the board is the primary evidence that learning is *visible*.

    python eval/plot_training.py --log logs/overfit/train.log --out outputs/phase4
    python eval/plot_training.py --board outputs/phase4/progression --out outputs/phase4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_RE = re.compile(
    r"epoch (?P<epoch>\d+) \| step (?P<step>\d+)/\d+ \| loss (?P<loss>[\d.]+) "
    r"\(avg (?P<avg>[\d.]+)\) \| lr (?P<lr>[\d.e+-]+) \| (?P<sit>[\d.]+)s/it"
)
VAL_RE = re.compile(r"\[validation\] epoch (?P<epoch>\d+) step (?P<step>\d+) \| val_loss (?P<val>[\d.]+)")
VRAM_RE = re.compile(r"vram (?P<alloc>[\d.]+)/(?P<peak>[\d.]+)GB")
GPU_RE = re.compile(r"gpu (?P<util>\d+)% (?P<temp>\d+)C")
GNORM_RE = re.compile(r"gnorm (?P<g>[\d.]+)")


def parse_log(path: Path) -> dict:
    train, val = [], []
    for line in path.read_text(errors="ignore").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP_RE.search(line)
        if m:
            rec = {k: float(v) for k, v in m.groupdict().items()}
            rec["step"], rec["epoch"] = int(rec["step"]), int(rec["epoch"])
            if (v := VRAM_RE.search(line)):
                rec["vram_alloc"], rec["vram_peak"] = float(v["alloc"]), float(v["peak"])
            if (g := GPU_RE.search(line)):
                rec["gpu_util"], rec["gpu_temp"] = float(g["util"]), float(g["temp"])
            if (n := GNORM_RE.search(line)):
                rec["gnorm"] = float(n["g"])
            train.append(rec)
            continue
        m = VAL_RE.search(line)
        if m:
            val.append({"epoch": int(m["epoch"]), "step": int(m["step"]), "val_loss": float(m["val"])})
    return {"train": train, "val": val}


def plot(data: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr, va = data["train"], data["val"]
    if not tr:
        raise SystemExit("no training steps parsed from the log")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    s = [r["step"] for r in tr]

    ax[0, 0].plot(s, [r["loss"] for r in tr], lw=0.9, alpha=0.45, label="loss (step)")
    ax[0, 0].plot(s, [r["avg"] for r in tr], lw=2.0, label="loss (running mean)")
    if va:
        ax[0, 0].plot([r["step"] for r in va], [r["val_loss"] for r in va],
                      "o-", ms=4, lw=1.5, color="crimson", label="val loss")
    ax[0, 0].set_xlabel("optimiser step"); ax[0, 0].set_ylabel("loss")
    ax[0, 0].set_title("Training / validation loss"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(s, [r["avg"] for r in tr], lw=2.0)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("optimiser step"); ax[0, 1].set_ylabel("loss (log)")
    ax[0, 1].set_title("Loss, log scale"); ax[0, 1].grid(alpha=0.3, which="both")

    ax[1, 0].plot(s, [r["lr"] for r in tr], lw=1.6, color="darkgreen")
    ax[1, 0].set_xlabel("optimiser step"); ax[1, 0].set_ylabel("learning rate")
    ax[1, 0].set_title("LR schedule"); ax[1, 0].grid(alpha=0.3)

    if any("gnorm" in r for r in tr):
        g = [(r["step"], r["gnorm"]) for r in tr if "gnorm" in r]
        ax[1, 1].plot([x for x, _ in g], [y for _, y in g], lw=1.0, color="purple")
        ax[1, 1].set_ylabel("grad norm")
    if any("vram_peak" in r for r in tr):
        ax2 = ax[1, 1].twinx()
        ax2.plot(s, [r.get("vram_peak", float("nan")) for r in tr], lw=1.2, color="grey", alpha=0.7)
        ax2.set_ylabel("peak VRAM (GB)", color="grey")
    ax[1, 1].set_xlabel("optimiser step")
    ax[1, 1].set_title("Gradient norm and VRAM"); ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    p = out / "training_curves.png"
    fig.savefig(p, dpi=110)
    return p


def board(prog_dir: Path, subset_csv: Path, root: Path, out: Path, panel_h: int = 300) -> Path:
    """Person | Garment | base | ...checkpoints... — one row per sample."""
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont

    def font(sz, bold=False):
        for c in [f"/usr/share/fonts/TTF/DejaVuSans{'-Bold' if bold else ''}.ttf",
                  f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]:
            if Path(c).exists():
                try:
                    return ImageFont.truetype(c, sz)
                except Exception:
                    pass
        return ImageFont.load_default()

    meta = json.loads((prog_dir / "progression.json").read_text())
    stages, ids = meta["stages"], meta["ids"]
    sub = pd.read_csv(subset_csv, dtype={"id": str}).set_index("id")
    f_lab, f_sm = font(16, True), font(13)

    def fit(im):
        return im.resize((max(1, int(im.width * panel_h / im.height)), panel_h), Image.LANCZOS)

    rows = []
    for sid in ids:
        panels = [(fit(Image.open(root / sub.loc[sid].person_path).convert("RGB")), "Person / GT"),
                  (fit(Image.open(root / sub.loc[sid].cloth_path).convert("RGB")), "Garment")]
        for st in stages:
            p = prog_dir / st / f"{sid}.png"
            if p.exists():
                panels.append((fit(Image.open(p).convert("RGB")), st))
        rows.append((sid, panels))

    pad, lab_h, cap_h = 8, 24, 20
    W = max(sum(p.width for p, _ in ps) + pad * (len(ps) + 1) for _, ps in rows)
    H = lab_h + len(rows) * (panel_h + cap_h + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    y = pad
    for i, (sid, panels) in enumerate(rows):
        x = pad
        for im, lab in panels:
            if i == 0:
                d.text((x, 2), lab, font=f_lab, fill=(20, 20, 20))
            canvas.paste(im, (x, y + lab_h if i == 0 else y))
            x += im.width + pad
        base_y = (y + lab_h if i == 0 else y) + panel_h
        title = str(sub.loc[sid].title)[:90] if "title" in sub.columns else ""
        d.text((pad, base_y + 3), f"{sid}   {title}", font=f_sm, fill=(110, 110, 110))
        y = base_y + cap_h + pad

    p = out / "progression_board.jpg"
    canvas.save(p, quality=90)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--board", default=None, help="progression dir")
    ap.add_argument("--subset", default="eval/overfit_subset.csv")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="outputs/phase4")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.log:
        data = parse_log(Path(args.log))
        (out / "curve_data.json").write_text(json.dumps(data, indent=2))
        p = plot(data, out)
        tr, va = data["train"], data["val"]
        first, last = tr[0]["avg"], tr[-1]["avg"]
        print(f"parsed {len(tr)} steps, {len(va)} validations -> {p}")
        print(f"  train loss: {first:.5f} -> {last:.5f}  ({100*(1-last/first):.1f}% reduction)")
        if va:
            print(f"  val   loss: {va[0]['val_loss']:.5f} -> {va[-1]['val_loss']:.5f}")
        if any("vram_peak" in r for r in tr):
            pk = max(r["vram_peak"] for r in tr if "vram_peak" in r)
            print(f"  peak VRAM: {pk:.2f} GB")
        if any("gpu_temp" in r for r in tr):
            temps = [r["gpu_temp"] for r in tr if "gpu_temp" in r]
            print(f"  GPU temp: {min(temps):.0f}-{max(temps):.0f} C")

    if args.board:
        p = board(Path(args.board), Path(args.subset), Path(args.root), out)
        print(f"board -> {p}")


if __name__ == "__main__":
    main()
