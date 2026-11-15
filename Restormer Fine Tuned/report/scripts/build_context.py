from build_common import *
from level_sections import NOTES
T=P['total']; FZ=P['frozen']; TRN=P['trainable']
C=f"""# PROJECT_CONTEXT.md — Restormer fine-tuning for defocus-blurred text (HDC 2021, Levels 0–9)

> **Purpose of this file.** A self-contained briefing for a person or an LLM that knows nothing about this project. Everything needed to discuss, extend, write up or debug the work is here: goal, data, model, exact training recipe, every headline number, known gaps, and where each file lives. Numbers were extracted from the project's own result files on 12 November 2026. When this file and a result CSV disagree, the CSV wins.

---

## 0. Quick facts

- **Project:** "Scene Text Recognition System" — Final Year Project (B.Tech CSE, Amrita Vishwa Vidyapeetham, course 23CSE498 Project Phase II, Team 48). Owner of this part: Praneeth (GitHub `PraneethUday`).
- **Research question:** does restoring defocus-blurred text images with a fine-tuned Restormer improve downstream OCR accuracy, and how does that change with blur severity?
- **What was done (Phase 2):** one Restormer model fine-tuned **per blur level** (Levels 0–9 of the HDC 2021 dataset), each evaluated on 40 held-out images with PSNR, SSIM and EasyOCR accuracy, against two baselines (blurred input; pretrained Restormer without fine-tuning) and a reference (OCR on the sharp ground truth).
- **One-line result:** fine-tuning raises PSNR by 2.4–8.4 dB and SSIM by 0.11–0.23 at every level; OCR character accuracy goes from 0% (EasyOCR detects nothing) to 96.9% at Level 5, but restoration slightly *hurts* OCR at Level 1 where the input is already readable, and accuracy falls to 52.8% at Level 9.
- **Compute:** Google Colab, Tesla T4 (15.6 GB), PyTorch; data and outputs on Google Drive under `FinalYearProject-Phase2/`.
- **Related repos:** final team repo `SUSHILV-30/Recognition-Aware-Blind-Deblurring-for-Degraded-Text-Images`; earlier Phase-1 work in `PraneethUday/Restromer-STR` (and `PraneethUday/WPIP`). Team paper: "Blind Deblurring of Degraded Text Images: A Comparative Analysis of Restoration Paradigms" (Paper ID 1110, ICMLDE 5.0); this work is the fine-tuned-Restormer part.

---

## 1. Status of every component

| Component | Status | Notes |
|:--|:--|:--|
| Levels 1–9 | **Complete** | Checkpoints, logs, split, registration, per-image metrics, OCR transcripts, restored images |
| Level 0 | **Trained, evaluation incomplete** | Checkpoints + restored images exist; no summary/per-image/OCR CSVs and no manifest. PSNR/SSIM in this file were recomputed from saved images (method validated to ≤0.003 dB on Levels 1–9). **No OCR numbers exist for Level 0.** |
| Level 5 | Complete, **legacy layout** | Trained first with an older notebook; results under `level_05/results/level5_results/`; manifest lacks config; no `train_log.txt`. σ ≈ 25 px, patch 128 × batch 4 inferred from notebook comments |
| Level 10 | **Not trained** | Dataset zip exists (`HDC2021_step_10.zip`); manifest says pending |
| Combined "ultimate" model (Levels 5–10 in one model) | **Started, not evaluated** | Preprocessing done 20 Oct 2026; only `ultimate_last.pth` exists; no metrics. Config in §9 |

---

## 2. Dataset

- **Source:** Helsinki Deblur Challenge 2021 (HDC 2021). Text on an e-ink display photographed by two cameras through a beam splitter. **CAM01 = in focus = ground truth. CAM02 = defocused = input.** "Focus step N" (called "Level N" here) is one fixed defocus amount; larger N = more blur. The blur is real optical defocus (disc-like PSF + sensor noise + slight magnification change), not synthetic.
- **Per level:** `stepN/{{Times,Verdana}}/{{CAM01,CAM02}}/focusStep_N_<timesR|verdanaRef>_size_30_sample_XXXX.tif` plus a `.txt` with the true text. 100 samples per font → **200 pairs per level**. Also `focusStep_N_PSF.tif`, `_LSF_X.tif`, `_LSF_Y.tif` calibration targets.
- **Images:** single-channel TIFF, 2360 × 1460 px; after registration and a 48 px trim → **2264 × 1364**. Each shows three lines of random mixed-case letters, 30–32 characters including spaces (so no language prior helps).
- **Split (identical sample IDs at every level, seed 42, per font):** 75 train + 5 val + 20 test per font → **150 train / 10 val / 40 test**. Stored in `results/split.json`.
- **Archives:** `HDC2021_step0.zip` … `HDC2021_step9.zip`, `HDC2021_step_10.zip`, ≈2.5 GB each.

---

## 3. Model: Restormer (defocus-deblurring configuration)

```python
Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4,6,6,8],
          num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66,
          bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
```

- Code: official repo `swz30/Restormer`, file `basicsr/models/archs/restormer_arch.py` (imported directly). Weights: `single_image_defocus_deblurring.pth` (104.7 MB).
- **Structure:** 3×3 conv patch embedding (3→48) → encoder L1 (4 blocks, 48 ch, full res) → down → encoder L2 (6 blocks, 96 ch, 1/2) → down → encoder L3 (6 blocks, 192 ch, 1/4) → down → latent (8 blocks, 384 ch, 1/8) → up + concat skip + 1×1 reduce → decoder L3 (6 blocks, 192 ch) → up + concat + 1×1 reduce → decoder L2 (6 blocks, 96 ch) → up + concat (no reduce) → decoder L1 (4 blocks, 96 ch) → refinement (4 blocks, 96 ch) → 3×3 conv (96→3) → **add input (global residual)**. Down = 3×3 conv halving channels + PixelUnshuffle(2); up = 3×3 conv doubling channels + PixelShuffle(2). Heads per stage: 1, 2, 4, 8 (decoder mirrors; decoder L1 and refinement use 1 head). 44 Transformer blocks in total.
- **Transformer block:** `X' = X + MDTA(LN(X)); X_out = X' + GDFN(LN(X'))`.
  - **MDTA** (multi-Dconv head transposed attention): 1×1 conv C→3C, 3×3 depth-wise conv, split into Q/K/V, reshape to heads×(C/heads)×HW, L2-normalise Q and K, attention = softmax(τ·QKᵀ) of size (C/heads)×(C/heads) with learned temperature τ per head, output A·V → 1×1 conv. Attention is across **channels**, so cost is linear in pixel count while still pooling over the whole image.
  - **GDFN** (gated-Dconv feed-forward): 1×1 conv C→2·⌊2.66C⌋, 3×3 depth-wise conv, split in two halves, `GELU(x1) * x2`, 1×1 conv back to C.
- Fully convolutional; input sides must be multiples of 8; grayscale input is replicated to 3 channels and the 3 output channels are averaged.
- **Parameters:** total {T:,} ({T/1e6:.2f} M). **Frozen during fine-tuning: {FZ:,} ({FZ/1e6:.2f} M)** = `patch_embed`, `encoder_level1`, `down1_2`, `encoder_level2`, `down2_3`, `encoder_level3`, `down3_4`. **Trainable: {TRN:,} ({TRN/1e6:.2f} M)** = `latent` (14.37 M), `up4_3`, `reduce_chan_level3`, `decoder_level3`, `up3_2`, `reduce_chan_level2`, `decoder_level2`, `up2_1`, `decoder_level1`, `refinement`, `output`.

{param_table()}
---

## 4. Training recipe (identical for every level; notebook `FinalYearProject_Phase2_Level1.ipynb`, only `LEVEL` changes)

1. **Load:** TIFF → float32 → mean over channels → divide by the image's 99.5th percentile → clip [0,1]. Sanity check that CAM01 has more edge energy than CAM02.
2. **Split:** as in §2; disjointness asserted.
3. **Estimate blur size σ:** blur the sharp image with Gaussian σ ∈ {{5,10,…,100}} px, template-match the centre region against the real input, pick best NCC; median over 4 training pairs. (5 px is the grid floor.)
4. **Register:** on 12 training pairs, match 320×160 px blurred-GT templates (80 px grid, ±60 px search, NCC ≥ 0.8, relaxed to 0.7/0.6 if < 50 matches) → RANSAC affine (3 px, 5000 iters) → **one global 2×3 affine per level** (`registration_affine.npy`, maps input coords → GT coords). Warp every GT into the input frame (bicubic), trim 48 px. Alignment score (NCC of blurred-GT vs input on text area) logged for all 200 pairs in `pairing_checks.csv`.
5. **Model:** build, load pretrained weights (strict), freeze encoder path (§3). `INIT_FROM='pretrained'` for all levels (the `'previous'` curriculum option exists but was not used).
6. **Patches:** random crop from a random training image, accepted only if the GT patch std > 0.05 (contains text; up to 20 tries); random 90° rotation and horizontal flip. **Patch/batch rule: σ ≤ 30 → 128 px × 4; σ ≤ 50 → 192 px × 2; else 256 px × 2.**
7. **Optimisation:** Charbonnier loss (ε = 1e-3); AdamW (lr 2e-5, weight decay 1e-4, betas 0.9/0.999); cosine annealing over 3000 iterations to 1e-7; **3000 iterations**; AMP float16 with GradScaler; gradient-norm clip 1.0; seed 42.
8. **Validation / selection:** every 500 iterations on the 10 validation images (full-image tiled inference); **best validation SSIM** checkpoint saved as `levelN_best.pth`; `levelN_last.pth` holds full resume state.
9. **Inference:** 768 px tiles, 192 px overlap (256 if σ > 50), linear feathering over the outer 96 px, reflect padding to a multiple of 8; output saved as 16-bit PNG.
10. **Evaluation:** PSNR and SSIM (`skimage`, data range 1, whole image) vs registered GT; EasyOCR (`Reader(['en'])`, default models, `detail=1, paragraph=False`), boxes grouped into lines by y, sorted by x, joined by spaces; character accuracy = max(0, 1 − Levenshtein/len(GT)); token accuracy = share of GT tokens found in the OCR tokens (order ignored). CER, WER, exact-match and empty-OCR rates were added post hoc from the saved transcripts.

Per-level training facts:

{train_table()}
Registration facts:

{reg_table()}
---

## 5. Results (test set, 40 images per level)

### 5.1 Headline

{headline_table()}
### 5.2 All methods and metrics

Methods: *Blurred input* (CAM02), *Pretrained Restormer* (no fine-tuning), *Fine-tuned Restormer* (this work), *Sharp GT* (EasyOCR on ground truth; image metrics undefined).

{summary_table()}
### 5.3 Gains and per-image consistency

{gains_table()}
### 5.4 By font (fine-tuned and baselines)

{font_table(range(10))}
---

## 6. Level-by-level findings

"""
for L in range(10): C+=f"### Level {L}\n\n"+NOTES[L].replace('(see §','(see REPORT.md §').replace('images in §3','images in REPORT.md §3')+"\n\n"
C+=f"""---

## 7. Interpretation — the points a discussion or write-up should get right

1. **Fine-tuning, not the architecture alone, produces the gain.** The pretrained model is never more than 0.05 dB above the blurred input and its OCR accuracy stays within 4 points of it.
2. **OCR is the primary outcome; PSNR/SSIM are secondary.** Highest fine-tuned PSNR (Level 1) coincides with an OCR *loss*; the largest OCR gain (Level 5, +96.9 points) comes with a moderate PSNR.
3. **"Sharp GT (upper bound)" is not a ceiling.** EasyOCR reads the sharp camera images at 90.8–94.1% character accuracy (it drops wide glyphs such as W and V), and the fine-tuned outputs at Levels 2–5 are read better than that (96.1–98.3%).
4. **There is a usefulness window.** Level 0–1: input already readable, restoration unnecessary for OCR. Levels 2–5: restoration brings OCR to ≥ 96%. Levels 6–8: 73–83% characters but only 22–32% whole tokens. Level 9: 53% characters, hallucinated strokes, unstable training.
5. **Serif vs sans-serif.** From Level 6 Times degrades much faster than Verdana (Level 9: 34.3% vs 71.3%).
6. **Confounds across levels.** Registration quality falls with blur (alignment 0.97 → 0.78), patch/batch changes at Level 7 (12,000 → 6,000 patches seen), σ estimates for Levels 0–2 are at the grid floor, and Level 6 used 128 px patches at σ ≈ 30 (patch ≈ 4σ).
7. **Scope.** Each model is trained and tested on its own level (level-matched, effectively non-blind). Cross-level generalisation is untested; the combined model was meant to address it and is unfinished.
8. **SSIM across levels is not monotonic for the input** (0.372 at Levels 5–7, 0.417–0.419 at Levels 8–9). Compare methods within a level.
9. **Statistics.** One training run per level, one seed. "±" values are spread over the 40 test images, not over repeated trainings.

---

## 8. Known gaps and suggested next steps

- Run the OCR/summary cells for **Level 0** (checkpoint exists; `RESUME=True`).
- Re-run **Level 6 with `PATCH_OVERRIDE=(192, 2)`**; re-run **Level 9 with more iterations** and/or `INIT_FROM='previous'`.
- Train **Level 10**.
- **Cross-level evaluation** of each per-level checkpoint.
- Finish and evaluate the **combined model**.
- Add a **second recogniser** (e.g. TrOCR) so OCR conclusions do not rest on EasyOCR alone.
- Earlier **Phase-1** results (a small "focusStep" subset for levels 0–4, ~113 pairs, different split and notebook; e.g. "+3.2 dB at Level 1") are **not comparable** with this file and are superseded by it. An earlier metric mix-up (Swin2SR-only numbers attributed to Restormer) was already corrected in the repos; do not reuse those figures.

---

## 9. Combined "ultimate" model (unfinished)

- Folder `ultimate_model/`. Levels 5–10 together, 200 pairs each, 150/10/40 split per level.
- Registration per level (σ / matches / RANSAC inliers / NCC before → after): L5 25 px / 2950 / 1581 / 0.792 → 0.892; L6 30 / 2834 / 915 / 0.758 → 0.867; L7 37 / 2584 / 485 / 0.722 → 0.836; L8 45 / 2405 / 249 / 0.669 → 0.804; L9 47 / 2060 / 148 / 0.626 → 0.780; L10 55 / 1745 / 82 / 0.559 → 0.689. Per-image refinement applied to 87 images at Level 5 and 20 at Level 6.
- Planned training: progressive patches — to iteration 2500 at 160 px × 4, to 5000 at 224 px × 2, to 6000 at 288 px × 2; LR 8e-5 with the encoder at half rate (encoder **not** frozen).
- Pretrained baseline at iteration 0: validation 14.78 dB / SSIM 0.4396.
- Only `checkpoints/ultimate_last.pth` (418.8 MB) exists. No best checkpoint, no later validation lines, no test metrics.

---

## 10. File map

**Drive export (`FinalYearProject-Phase2/`)**

```
pretrained/single_image_defocus_deblurring.pth
all_levels_summary.csv, all_levels_trend.png            (Levels 1–9)
level_NN/
  manifest.json                                         (absent for level_00)
  checkpoints/levelN_best.pth (104.7 MB), levelN_last.pth (283.7 MB)
  logs/train_log.txt, training_history.json             (level_05: none in logs/)
  results/                                              (level_05: results/level5_results/)
    split.json, registration_affine.npy, pairing_checks.csv
    levelN_summary.csv, levelN_per_image.csv, levelN_by_font.csv, ocr_outputs.csv   (absent for level_00)
    restored/pretrained/<id>.png, restored/finetuned/<id>.png   (16-bit, 40 each)
    figures/training_curves.png, levelN_comparison.png
ultimate_model/{{cache,logs,results,checkpoints}}
```

**This report folder (`Restormer_Report_Levels0-9/`)**

```
REPORT.md / .docx / .pdf / .html      full report
PROJECT_CONTEXT.md                    this file
tables/                               COMPARISON_TABLES.md + CSVs (summary, per-image, by-font, gains, training/registration, OCR transcripts)
comparisons/level_NN_comparison.jpg   blurred | fine-tuned | ground truth, 4 test images, with metrics and OCR text
comparisons/level_NN_closeup.jpg      native-resolution centre crops
images/level_NN/                      individual panels: L<N>_<id>_{{blurred,pretrained,finetuned,groundtruth}}.jpg
figures/                              restormer_architecture.(png|svg), transformer_block_detail.png, pipeline_overview.png, charts
source_results/                       copies of the original CSVs, logs, manifests, notebook figures
scripts/                              scripts used to build tables and figures
```

Image IDs look like `timesR_0043` / `verdanaRef_0058` (font + 4-digit sample number). The four images shown in every comparison grid are test-split positions 1, 11, 21, 31: `timesR_0043`, `timesR_0074`, `verdanaRef_0058`, `verdanaRef_0086`.

---

## 11. Glossary

CAM01/CAM02: sharp/defocused camera. Level/step: defocus setting. σ: Gaussian-equivalent blur width used for matching. NCC: normalised cross-correlation (alignment score). MDTA/GDFN: Restormer attention / feed-forward sub-layers. Charbonnier: smooth L1 loss. AMP: mixed precision. CER/WER: character/word error rate. Token accuracy: bag-of-words recall of GT tokens. "Sharp GT (upper bound)": EasyOCR on the ground-truth image.

---

## 12. Guidance for an LLM using this file

- Quote numbers from §5; do not interpolate or invent values for Level 0 OCR, Level 10, or the combined model — they do not exist.
- Keep the distinction between *blurred input*, *pretrained* and *fine-tuned* rows; gains are always stated relative to the blurred input unless "vs pretrained" is said.
- Do not describe the per-level models as blind deblurring; each is level-matched.
- Parameter split is **3.79 M frozen / 22.34 M trainable** in this Phase-2 notebook (older project notes with other figures refer to a different freezing scheme in Phase 1).
- The owner prefers complete outputs over diffs, explanations in two registers (technical, then plain-language), and claims that can be checked against the CSVs.
"""
open(f'{OUT}/PROJECT_CONTEXT.md','w').write(C); print(len(C))
