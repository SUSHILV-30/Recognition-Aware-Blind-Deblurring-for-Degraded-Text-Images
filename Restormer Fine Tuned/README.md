# Restormer-STR

A staged pipeline for reading text off **defocus-blurred document pages** — the
case where OCR fails not because the layout is hard but because the glyph edges
have been destroyed by the camera's focus.

```
 blurred page ──▶ Restormer ──▶ EasyOCR ──▶ TrOCR ──▶ transcript
                  (deblur)      (detect)    (read)
```

Final year project. Built on the [focusStep](docs/dataset.md) corpus of
synthetically defocused text pages.

---

## Why staged, and why these models

Restoration and recognition are usually studied apart. The premise here is that
on blurred documents the restoration stage is not cosmetic — it decides whether
the recognition stage has anything to work with at all. Splitting the pipeline
into switchable stages makes that contribution measurable rather than assumed.

| Stage | Model | What it fixes |
|---|---|---|
| 1. Restoration | Restormer, fine-tuned on focusStep | Removes defocus blur; recovers glyph edges |
| 2. Detection | EasyOCR (detector only, `recognizer=False`) | Locates text regions |
| 3. Recognition | `microsoft/trocr-base-printed` | Decodes each region to a string |

<p align="center">
  <img src="docs/images/restormer_architecture_diagram.png" width="100%" alt="Restormer architecture: 4-level U-Net with MDTA and GDFN transformer blocks">
</p>

Two findings drove the model choices:

- **Off-the-shelf restoration weights do not transfer.** The official
  `single_image_defocus_deblurring.pth` checkpoint is trained on DPDD natural
  scenes. Applied unchanged to text pages it gave no benefit and slightly
  *degraded* PSNR — dense high-frequency glyph edges are not natural image
  statistics. Fine-tuning on paired focusStep data is what makes the stage
  worth its compute.
- **No separate super-resolution stage.** An earlier version of this pipeline
  ran a Swin2SR 4× upscale between restoration and detection. It's been
  dropped: EasyOCR/TrOCR perform detection and recognition directly on
  Restormer's full-resolution output, and keeping Swin2SR in the diagram was
  actively misleading — the numbers historically reported as "restoration
  quality" in this repo were measured on Swin2SR upscaling the *raw blurred*
  page, with Restormer never in that run (see **Results** below). Removing
  the stage removes that confound instead of leaving it in the architecture
  diagram unexplained.

<p align="center">
  <img src="docs/images/finetuned_comparison_bar.png" width="100%" alt="PSNR and SSIM at each defocus level for input, pretrained Restormer, and fine-tuned Restormer">
</p>

Fine-tuning is the difference between a stage that helps and one that doesn't:
the pretrained checkpoint tracks the unrestored input almost exactly, while
fine-tuning on focusStep lifts PSNR by up to +3.2 dB and SSIM by up to +0.18,
with the gap widening as blur gets worse. This comparison is Restormer against
itself (fine-tuned vs. pretrained), measured independently of the OCR stages
below.

---

## Results

<p align="center">
  <img src="docs/images/blur_level_progression.png" width="100%" alt="The same text patch at ground truth and defocus levels 1 through 4, showing increasing blur">
</p>

### Fine-tuned vs. pretrained Restormer, by blur level

This part is a real, verified Restormer result. Input, pretrained-checkpoint
output, fine-tuned output, and ground truth, side by side at each defocus
level:

<p align="center">
  <img src="docs/images/level_1_finetuned_vs_pretrained.png" width="100%" alt="Defocus level 1 comparison: input, pretrained, fine-tuned, and ground truth">
</p>
<p align="center">
  <img src="docs/images/level_2_finetuned_vs_pretrained.png" width="100%" alt="Defocus level 2 comparison: input, pretrained, fine-tuned, and ground truth">
</p>
<p align="center">
  <img src="docs/images/level_3_finetuned_vs_pretrained.png" width="100%" alt="Defocus level 3 comparison: input, pretrained, fine-tuned, and ground truth">
</p>
<p align="center">
  <img src="docs/images/level_4_finetuned_vs_pretrained.png" width="100%" alt="Defocus level 4 comparison: input, pretrained, fine-tuned, and ground truth">
</p>

The fine-tuned output stays legible well past the point where the pretrained
checkpoint's gains over the raw input flatten out — visible confirmation of
the +3.2 dB / +0.18 gap above. This comparison was run on Restormer's own
output against sharp ground truth (`CAM01_focused`), independent of any
downstream OCR stage.

### Previous baseline — superseded, not a Restormer result

Earlier versions of this README quoted a headline table (SSIM 0.7320, PSNR
31.77 dB, 51.0% token exact-match, 83.3% character accuracy) as if it
described the pipeline above. It doesn't, and attributing it to Restormer
would just repeat the mistake this rewrite exists to fix. To be precise about
what that table actually measured:

- It came from **Swin2SR → EasyOCR → TrOCR**, run with `--no-restore` —
  Restormer was not in that pipeline invocation.
- Its SSIM/PSNR compared Swin2SR's output to the *bicubically upscaled raw
  input*, not to sharp ground truth — so it measured how much Swin2SR changed
  the image, not restoration quality. That's also why those scores rose
  slightly as blur got worse, which is backwards for a fidelity metric.
- Only 56 pages, blur levels 2–4, were in that evaluation subset — levels 0–1
  were never covered.

Full breakdown of why those numbers don't mean what they look like they mean:
[docs/results.md](docs/results.md) (kept as-is, unedited, for the record).

### What's actually still open

The one comparison that would justify this architecture — full pipeline
(Restormer → EasyOCR → TrOCR) evaluated against the Swin2SR-only baseline
above, on the same 56 pages — has **not been run end-to-end in this repo**.
`scripts/evaluate.py` has been updated to score Restormer's output against
sharp ground truth once that run happens (see **Layout** below). That's the
next experiment, and it's the one that decides whether this project's premise
holds.

---

## Phase 2 — one fine-tuned Restormer per HDC level, Levels 0–9

Phase 2 fine-tunes the DPDD-pretrained Restormer separately for every HDC 2021
focus step from 0 to 9, with one notebook in which only `LEVEL` changes:
[`notebooks/05_phase2_per_level_0_to_9.ipynb`](notebooks/05_phase2_per_level_0_to_9.ipynb).
Each run registers CAM01 to CAM02 with one global affine (blur-matched template
matching + RANSAC), uses the same 150 / 10 / 40 train / val / test split at every
level (seed 42), freezes the encoder (3.79 M frozen, 22.34 M trainable), trains
3,000 iterations of Charbonnier loss with AdamW at 2×10⁻⁵, keeps the checkpoint
with the best validation SSIM, and scores the 40 test pages with PSNR, SSIM and
EasyOCR. Patch size grows with the estimated blur: 128 px × 4 up to σ = 30 px,
192 px × 2 above that.

Test set, 40 pages per level (`results/phase2/all_levels_summary.csv`, Level 0
recomputed from the saved outputs):

| Level | σ (px) | PSNR blurred → fine-tuned | SSIM blurred → fine-tuned | OCR char. acc. blurred → fine-tuned | Sharp GT |
|--:|--:|--:|--:|--:|--:|
| 0 | 5 | 26.08 → **28.74** | 0.524 → **0.691** | not run | — |
| 1 | 5 | 23.17 → **28.87** | 0.419 → **0.627** | 98.4% → 92.6% | 90.8% |
| 2 | 5 | 19.37 → **27.38** | 0.390 → **0.615** | 94.2% → **96.1%** | 92.2% |
| 3 | 12 | 17.34 → **25.72** | 0.380 → **0.606** | 75.5% → **98.3%** | 92.5% |
| 4 | 20 | 16.07 → **24.10** | 0.373 → **0.595** | 23.5% → **97.4%** | 91.3% |
| 5 | 25 | 15.35 → **21.98** | 0.372 → **0.582** | 0.0% → **96.9%** | 94.1% |
| 6 | 30 | 15.03 → **20.19** | 0.372 → **0.569** | 0.0% → **83.2%** | 93.4% |
| 7 | 37 | 14.71 → **19.87** | 0.373 → **0.567** | 0.0% → **77.0%** | 91.6% |
| 8 | 45 | 14.45 → **19.47** | 0.417 → **0.563** | 0.0% → **73.5%** | 92.6% |
| 9 | 47 | 14.39 → **16.81** | 0.419 → **0.531** | 0.0% → **52.8%** | 93.2% |

Fine-tuning improves PSNR and SSIM on all 40 test images at every level. OCR
gains run from Level 2 onward and peak at Level 3; at Level 1 the blurred page is
already readable and restoration lowers OCR accuracy by 5.7 points. The
pretrained model without fine-tuning never moves more than 0.05 dB above the
input. Level 10 and the combined Levels 5–10 model are not trained or
evaluated. The full write-up, with every table and figure, is in
[`report/REPORT.md`](report/REPORT.md) ([PDF](report/REPORT.pdf)); a
self-contained briefing is in [`report/PROJECT_CONTEXT.md`](report/PROJECT_CONTEXT.md).

Earlier notebooks, kept for the record:

| Notebook | What it does |
|---|---|
| `02_phase2_level5_restormer_finetune.ipynb` | First Level 5 run, with outputs (the Level 5 results above come from this run) |
| `03_phase2_any_level_5_to_10.ipynb` | The per-level recipe for Levels 5–10, superseded by notebook 05 |
| `04_phase2_ultimate_all_levels_5_to_10.ipynb` | One model for Levels 5–10. Preprocessing ran; training was started but never evaluated (`results/phase2/ultimate_model/`) |

The Phase 1 focusStep results above (Levels 1–4, about 113 pairs, a different
split and notebook) are not comparable with Phase 2 and are superseded by it.

### Not in this repository

These are on the team Google Drive under `FinalYearProject-Phase2/` and are too
large for git. Every number in `results/phase2/` was computed from them.

| What | Where on Drive | Size |
|---|---|--:|
| Pretrained weights | `pretrained/single_image_defocus_deblurring.pth` | 105 MB |
| Per-level checkpoints | `level_NN/checkpoints/levelN_best.pth`, `levelN_last.pth` | 4.4 GB total |
| Restored test pages (16-bit PNG) | `level_NN/results/restored/{pretrained,finetuned}/` | 4.0 GB total |
| Combined-model checkpoint | `ultimate_model/checkpoints/ultimate_last.pth` | 419 MB |
| HDC 2021 dataset, steps 0–10 | `HDC2021_step*.zip` | 27.5 GB |

---

## Layout

```
├── src/
│   ├── models/restormer.py              Restormer architecture (MDTA + GDFN, 4-level U-Net)
│   ├── data/focusstep.py                Paired dataset loader, exact-filename matching
│   ├── restoration/
│   │   ├── finetune_restormer.py        Fine-tuning on focusStep
│   │   └── restore.py                   Tiled full-page inference
│   ├── recognition/ocr.py               EasyOCR detection + TrOCR recognition
│   └── pipeline.py                      End-to-end CLI, restoration ablation flag
├── scripts/
│   ├── evaluate.py                      Restored-vs-ground-truth SSIM/PSNR + token/character accuracy → CSV
│   └── ocr_eval.py                      EasyOCR scoring of any folder of restored pages
├── notebooks/
│   ├── 01_swin2sr_trocr_pipeline.ipynb  Original exploratory run (Swin2SR baseline), outputs retained for reference
│   ├── 02_phase2_level5_restormer_finetune.ipynb   First Phase 2 run, Level 5 (outputs kept)
│   ├── 03_phase2_any_level_5_to_10.ipynb           Earlier per-level recipe for Levels 5–10
│   ├── 04_phase2_ultimate_all_levels_5_to_10.ipynb Single model across Levels 5–10 (not evaluated)
│   └── 05_phase2_per_level_0_to_9.ipynb            Phase 2 per-level training and evaluation, Levels 0–9
├── results/
│   ├── phase2/
│   │   ├── all_levels_summary.csv, all_levels_trend.png
│   │   ├── levelN/                      N = 0–9: summary, per-image, by-font, OCR transcripts,
│   │   │                                pairing checks, split, registration affine, training log
│   │   │                                and history, manifest, training-curve and comparison figures
│   │   ├── level10/manifest.json        Level 10: not trained
│   │   └── ultimate_model/              Combined-model preprocessing: registration, splits, log
│   └── ocr/levels_1_4/                  Phase 1 OCR baselines (focusStep protocol)
├── report/                              Phase 2 report: REPORT.md/.pdf/.html, PROJECT_CONTEXT.md,
│                                        tables/ (every number as CSV), figures/, comparisons/,
│                                        scripts/ (Python that built the tables and figures)
└── docs/                                setup, dataset, methodology, results
`src/enhancement/swin2sr.py` is no longer called by `pipeline.py` but is left
in the repo rather than deleted, since the original exploratory notebook still
references it.

## Quick start

```bash
pip install -r requirements.txt

# Fine-tune the restoration stage
python -m src.restoration.finetune_restormer \
    --blurred-dir dataset/focusStep/blurred \
    --focused-dir dataset/focusStep/CAM01_focused \
    --pretrained weights/single_image_defocus_deblurring.pth \
    --epochs 30 --freeze-encoder

# Run the full pipeline
python -m src.pipeline \
    --input-dir dataset/raw --output-dir outputs/full \
    --restormer-weights checkpoints/restormer_focusstep_best.pth \
    --save-intermediate

# Ablation: same pipeline without deblurring
python -m src.pipeline --input-dir dataset/raw --output-dir outputs/no_restore --no-restore

# Score restoration quality (restored vs. sharp ground truth) and recognition accuracy
python scripts/evaluate.py \
    --restored-dir outputs/full/restored \
    --gt-image-dir dataset/focusStep/CAM01_focused \
    --pred-dir outputs/full/text --gt-dir dataset/raw
```

Full instructions, including where to put the pretrained weights and the
dataset: [docs/setup.md](docs/setup.md).

## Data is not in this repository

`dataset/` and `dataset_results_*/` total ~2.1 GB and are gitignored. The
directory layout the code expects is documented in
[docs/dataset.md](docs/dataset.md).

## Status

| Component | State |
|---|---|
| Restormer architecture, fine-tuning, and tiled inference | Fine-tuned on focusStep; fine-tuned-vs-pretrained comparison run and verified (+3.2 dB PSNR, +0.18 SSIM) |
| EasyOCR (detect) + TrOCR (read) | Code complete, previously run end-to-end in the Swin2SR-baseline configuration |
| Full pipeline: Restormer → EasyOCR → TrOCR | Wired together in `src/pipeline.py`; **not yet executed end-to-end in this repo** |
| `scripts/evaluate.py` restoration scoring | Updated to compare restored output against sharp ground truth; **not yet run**, since the full pipeline hasn't produced restored output yet |

The honest reading of the current state: Restormer's own deblurring gain is
verified in isolation, and the OCR stages are verified in isolation (on a
different, non-deblurred input). What's still missing is the one run that
connects them — Restormer's output going into EasyOCR/TrOCR — which is the
actual claim this project is built around.

## References

1. Zamir et al. *Restormer: Efficient Transformer for High-Resolution Image Restoration.* CVPR 2022. [arXiv:2111.09881](https://arxiv.org/abs/2111.09881)
2. Li et al. *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.* AAAI 2023. [arXiv:2109.10282](https://arxiv.org/abs/2109.10282)
3. Shi et al. *Robust Scene Text Recognition with Automatic Rectification (RARE).* CVPR 2016. [arXiv:1603.03915](https://arxiv.org/abs/1603.03915)
4. Alshawi, Tanha, Balafar. *An Attention-Based Convolutional Recurrent Neural Network for Scene Text Recognition.* IEEE Access, 2024. [doi:10.1109/ACCESS.2024.3352748](https://doi.org/10.1109/ACCESS.2024.3352748)
