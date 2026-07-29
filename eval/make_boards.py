#!/usr/bin/env python3
"""
Build visual comparison boards from Phase 2 baseline outputs.

Produces:
  boards/individual/<id>.jpg   Person | Garment | segfree output | masked output
                               with a caption carrying the sample's metadata
                               and metrics, so each board is self-describing.
  boards/contact_<axis>.jpg    Contact sheets grouped by a stratification axis,
                               for fast scanning of failure modes across strata.

Usage:
    python eval/make_boards.py --out-dir outputs/phase2
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PANEL_H = 480
PAD = 12
CAPTION_H = 54
LABEL_H = 26
BG = (255, 255, 255)
FG = (20, 20, 20)
MUTED = (110, 110, 110)


def font(sz, bold=False):
    cands = [
        "/usr/share/fonts/TTF/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/noto/NotoSans-%s.ttf" % ("Bold" if bold else "Regular"),
    ]
    for c in cands:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, sz)
            except Exception:
                pass
    return ImageFont.load_default()


F_LAB = font(15, bold=True)
F_CAP = font(14)
F_SM = font(12)


def fit(im: Image.Image, h: int) -> Image.Image:
    return im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)


def build_individual(out_root: Path, sid: str, meta: dict, boards: Path):
    panels = []
    for mode, label in [(None, "Person (input / GT)"), (None, "Garment (input)"),
                        ("segfree", "Output — segfree"), ("masked", "Output — masked")]:
        if mode is None:
            src = out_root / "segfree" / sid / ("person.jpg" if not panels else "garment.jpg")
            if not src.exists():
                src = out_root / "masked" / sid / ("person.jpg" if not panels else "garment.jpg")
        else:
            src = out_root / mode / sid / "output.png"
        if src.exists():
            panels.append((fit(Image.open(src).convert("RGB"), PANEL_H), label))
        else:
            ph = Image.new("RGB", (int(PANEL_H * 0.66), PANEL_H), (238, 238, 238))
            ImageDraw.Draw(ph).text((10, PANEL_H // 2), "missing", font=F_CAP, fill=MUTED)
            panels.append((ph, label + " (missing)"))

    W = sum(p.width for p, _ in panels) + PAD * (len(panels) + 1)
    H = LABEL_H + PANEL_H + PAD * 2 + CAPTION_H
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    x = PAD
    for im, label in panels:
        d.text((x, 4), label, font=F_LAB, fill=FG)
        canvas.paste(im, (x, LABEL_H))
        x += im.width + PAD

    y = LABEL_H + PANEL_H + PAD
    line1 = (f"{sid}   {meta.get('source','')}   {meta.get('fabric_family','')}   "
             f"{meta.get('target_type','')}   clean={meta.get('is_clean','')}   "
             f"gscore={meta.get('garment_score',0):.4f} ({meta.get('gscore_q','')})   "
             f"cloth={meta.get('garment_photo_type','')}   res={meta.get('res_bucket','')}")
    d.text((PAD, y), line1, font=F_CAP, fill=FG)
    m = meta.get("_metrics", {})
    if m:
        parts = []
        for mode in ["segfree", "masked"]:
            if mode in m:
                s = m[mode]
                seg = f"{mode}: SSIM {s.get('ssim',float('nan')):.3f}  PSNR {s.get('psnr',float('nan')):.1f}dB"
                if "lpips" in s:
                    seg += f"  LPIPS {s['lpips']:.3f}"
                parts.append(seg)
        d.text((PAD, y + 19), "     |   ".join(parts), font=F_CAP, fill=(150, 30, 30))
    title = str(meta.get("title", ""))[:120]
    d.text((PAD, y + 37), title, font=F_SM, fill=MUTED)

    boards.mkdir(parents=True, exist_ok=True)
    canvas.save(boards / f"{sid}.jpg", quality=92)


def build_contact(out_root: Path, df: pd.DataFrame, axis: str, boards: Path, per_row=6, h=190):
    """Contact sheet: rows grouped by axis level, each cell = person|garment|segfree|masked."""
    groups = [(lv, g) for lv, g in df.groupby(axis, sort=True)]
    cells, headers = [], []
    for lv, g in groups:
        headers.append((str(lv), len(cells)))
        for r in g.itertuples():
            strip = []
            for src in [out_root / "segfree" / r.id / "person.jpg",
                        out_root / "segfree" / r.id / "garment.jpg",
                        out_root / "segfree" / r.id / "output.png",
                        out_root / "masked" / r.id / "output.png"]:
                if src.exists():
                    strip.append(fit(Image.open(src).convert("RGB"), h))
            if not strip:
                continue
            w = sum(s.width for s in strip) + 3 * 2
            c = Image.new("RGB", (w, h), BG)
            x = 0
            for s in strip:
                c.paste(s, (x, 0))
                x += s.width + 2
            cells.append((c, r.id))

    if not cells:
        return
    cw = max(c.width for c, _ in cells) + 8
    rows = (len(cells) + per_row - 1) // per_row
    hdr = 24
    H = rows * (h + 18) + len(headers) * hdr + 40
    W = per_row * cw + 20
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), f"Phase 2 baseline — grouped by {axis}   "
                    f"[person | garment | segfree | masked]", font=F_LAB, fill=FG)

    y = 34
    idx = 0
    hd = dict((i, lv) for lv, i in headers)
    while idx < len(cells):
        if idx in hd:
            d.text((10, y), f"── {hd[idx]} ──", font=F_LAB, fill=(40, 70, 140))
            y += hdr
        for col in range(per_row):
            if idx >= len(cells) or (idx in hd and col > 0):
                break
            c, sid = cells[idx]
            canvas.paste(c, (10 + col * cw, y))
            d.text((10 + col * cw, y + h + 2), sid, font=F_SM, fill=MUTED)
            idx += 1
        y += h + 18
    boards.mkdir(parents=True, exist_ok=True)
    canvas.save(boards / f"contact_{axis}.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/phase2")
    ap.add_argument("--boards", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    boards = Path(args.boards) if args.boards else out_root / "boards"
    runlog = pd.read_csv(out_root / "runlog.csv", dtype={"id": str})

    mpath = out_root / "metrics.csv"
    metrics = {}
    if mpath.exists():
        md = pd.read_csv(mpath, dtype={"id": str})
        for r in md.itertuples():
            metrics.setdefault(r.id, {})[r.mode] = {
                k: getattr(r, k) for k in ["ssim", "psnr", "lpips"] if hasattr(r, k)
            }

    ok = runlog[runlog.ok].drop_duplicates("id")
    print(f"Building {len(ok)} individual boards...")
    for r in ok.itertuples():
        meta = r._asdict()
        meta["_metrics"] = metrics.get(r.id, {})
        build_individual(out_root, r.id, meta, boards / "individual")

    for axis in ["source", "fabric_family", "gscore_q", "res_bucket", "target_type", "is_clean"]:
        if axis in ok.columns:
            print(f"Building contact sheet: {axis}")
            build_contact(out_root, ok, axis, boards)

    print(f"\nBoards written to {boards}")


if __name__ == "__main__":
    main()
