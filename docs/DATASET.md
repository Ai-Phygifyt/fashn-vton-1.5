# Dataset

## Layout

```
dataset/
├── train/
│   ├── image/<id>.jpg      person wearing the saree
│   ├── cloth/<id>.jpg      flat saree
│   └── pairs.txt           "<id>.jpg <id>.jpg" per line
├── val/
│   ├── image/<id>.jpg
│   ├── cloth/<id>.jpg
│   └── pairs.txt
├── train.csv               index consumed by the training pipeline
├── val.csv
├── metadata.csv            provenance and quality fields for every sample
├── rejected.csv            samples removed during cleaning, with reasons
└── summary.json            filter-by-filter counts
```

`image/<id>.jpg` and `cloth/<id>.jpg` share the same id — that is the pairing. Both
`pairs.txt` (VITON-HD convention) and the CSV index describe the same set; the CSV is what
the `datasets.dataset` loader reads, `pairs.txt` is provided for compatibility with other
VTON codebases.

## Ground truth

There is no separate `gt/` directory. In paired virtual try-on, `image/<id>.jpg` **is** the
ground truth — it is a photograph of the person wearing the target garment. The model input
is derived from it at load time by masking out the garment region (the clothing-agnostic
image), and the unmodified photograph is the reconstruction target. A duplicate `gt/` would
double the dataset size for no additional information.

## How this set was produced

Starting from 14,069 scraped pairs, filters were applied in order:

| Filter | Removed | Remaining |
|---|---:|---:|
| Starting corpus | — | 14,069 |
| Duplicate garments (perceptual hash) | 144 | 13,925 |
| Below resolution threshold | 666 | 13,259 |
| Not a genuine full garment | 5,185 | 8,074 |
| Garment image is a person, not a flat garment | 3,304 | 4,770 |
| Blouse-piece product render supplied instead of the saree | 2,503 | 2,267 |
| Frontal full-body standing pose required (max confidence) | 201 | 2,066 |
| Integrity + byte-level duplicate verification | 0 | 2,066 |
| Trimmed to target by garment quality | 66 | **2,000** |

Two of these deserve explanation, because neither is detectable from the dataset's own
quality flags:

**Garment image is a person.** A substantial fraction of listings supply a second on-model
photograph where a flat garment is expected. Pairing that with a person wearing the saree
gives the model the wrong input type.

**Blouse-piece renders.** Saree listings routinely sell a matching blouse piece and
photograph it separately, often as a standardised mockup recoloured per product. These were
detected via human-parsing signature (a `top` covering 10–35 % of the frame with no
`dress`/`skirt`/`pants` present) and verified by manual inspection: of 24 flagged images
checked, 24 were confirmed blouse renders.

## Validation

```bash
python scripts/validate_dataset.py --root dataset
```

Confirms every `pairs.txt` entry resolves, all images decode with valid dimensions, there
are no orphan files or duplicate entries, `image/` and `cloth/` id sets agree, and no ids
leak between splits.

## Rebuilding

```bash
python scripts/build_handoff_dataset.py --out dataset --target 2000
```

Requires the original scraped corpus and the cleaning index. Deterministic: the same inputs
produce the same split (seed 20260729).

## Known limitations

- **Rare undetected defects.** Fabric swatches with burnt-in captions, technical line
  drawings of blouses, and multi-garment collage images are not caught automatically.
  Estimated at ~2 % of samples. A small supervised classifier would close this.
- **Source concentration.** A minority of retailers contribute a disproportionate share.
  `metadata.csv` carries a `source` column for re-weighting or stratified sampling.
- **Mannequin subjects** are included and flagged via `target_type`. They are
  out-of-distribution relative to photographic subjects; filter on
  `target_type == "model"` if photoreal output matters.
- **Licensing.** Images are copyrighted product photography collected for research and model
  training. Source URLs are preserved per row in `metadata.csv`. Not for public
  redistribution.
