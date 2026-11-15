# PROJECT_CONTEXT.md — Restormer fine-tuning for defocus-blurred text (HDC 2021, Levels 0–9)

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
- **Per level:** `stepN/{Times,Verdana}/{CAM01,CAM02}/focusStep_N_<timesR|verdanaRef>_size_30_sample_XXXX.tif` plus a `.txt` with the true text. 100 samples per font → **200 pairs per level**. Also `focusStep_N_PSF.tif`, `_LSF_X.tif`, `_LSF_Y.tif` calibration targets.
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
- **Parameters:** total 26,126,644 (26.13 M). **Frozen during fine-tuning: 3,789,484 (3.79 M)** = `patch_embed`, `encoder_level1`, `down1_2`, `encoder_level2`, `down2_3`, `encoder_level3`, `down3_4`. **Trainable: 22,337,160 (22.34 M)** = `latent` (14.37 M), `up4_3`, `reduce_chan_level3`, `decoder_level3`, `up3_2`, `reduce_chan_level2`, `decoder_level2`, `up2_1`, `decoder_level1`, `refinement`, `output`.

| Module (name in code) | What it is | Parameters | % of model | Fine-tuning status |
|:--|:--|--:|--:|:--|
| `patch_embed` | 3×3 conv, 3→48 | 1,296 | 0.00 | Frozen |
| `encoder_level1` | 4 blocks, C=48, 1 head, full res | 125,116 | 0.48 | Frozen |
| `down1_2` | 3×3 conv 48→24 + PixelUnshuffle(2) → 96 ch | 10,368 | 0.04 | Frozen |
| `encoder_level2` | 6 blocks, C=96, 2 heads, 1/2 res | 707,232 | 2.71 | Frozen |
| `down2_3` | 3×3 conv 96→48 + PixelUnshuffle(2) → 192 ch | 41,472 | 0.16 | Frozen |
| `encoder_level3` | 6 blocks, C=192, 4 heads, 1/4 res | 2,738,112 | 10.48 | Frozen |
| `down3_4` | 3×3 conv 192→96 + PixelUnshuffle(2) → 384 ch | 165,888 | 0.63 | Frozen |
| `latent` | 8 blocks, C=384, 8 heads, 1/8 res | 14,370,448 | 55.00 | Trainable |
| `up4_3` | 3×3 conv 384→768 + PixelShuffle(2) → 192 ch | 2,654,208 | 10.16 | Trainable |
| `reduce_chan_level3` | 1×1 conv 384→192 after concat | 73,728 | 0.28 | Trainable |
| `decoder_level3` | 6 blocks, C=192, 4 heads, 1/4 res | 2,738,112 | 10.48 | Trainable |
| `up3_2` | 3×3 conv 192→384 + PixelShuffle(2) → 96 ch | 663,552 | 2.54 | Trainable |
| `reduce_chan_level2` | 1×1 conv 192→96 after concat | 18,432 | 0.07 | Trainable |
| `decoder_level2` | 6 blocks, C=96, 2 heads, 1/2 res | 707,232 | 2.71 | Trainable |
| `up2_1` | 3×3 conv 96→192 + PixelShuffle(2) → 48 ch | 165,888 | 0.63 | Trainable |
| `decoder_level1` | 4 blocks, C=96 (48+48 concat), 1 head, full res | 471,484 | 1.80 | Trainable |
| `refinement` | 4 blocks, C=96, 1 head, full res | 471,484 | 1.80 | Trainable |
| `output` | 3×3 conv 96→3, then + input | 2,592 | 0.01 | Trainable |
| **Total** |  | **26,126,644** | 100.00 | Frozen 3,789,484 · Trainable 22,337,160 |

---

## 4. Training recipe (identical for every level; notebook `FinalYearProject_Phase2_Level1.ipynb`, only `LEVEL` changes)

1. **Load:** TIFF → float32 → mean over channels → divide by the image's 99.5th percentile → clip [0,1]. Sanity check that CAM01 has more edge energy than CAM02.
2. **Split:** as in §2; disjointness asserted.
3. **Estimate blur size σ:** blur the sharp image with Gaussian σ ∈ {5,10,…,100} px, template-match the centre region against the real input, pick best NCC; median over 4 training pairs. (5 px is the grid floor.)
4. **Register:** on 12 training pairs, match 320×160 px blurred-GT templates (80 px grid, ±60 px search, NCC ≥ 0.8, relaxed to 0.7/0.6 if < 50 matches) → RANSAC affine (3 px, 5000 iters) → **one global 2×3 affine per level** (`registration_affine.npy`, maps input coords → GT coords). Warp every GT into the input frame (bicubic), trim 48 px. Alignment score (NCC of blurred-GT vs input on text area) logged for all 200 pairs in `pairing_checks.csv`.
5. **Model:** build, load pretrained weights (strict), freeze encoder path (§3). `INIT_FROM='pretrained'` for all levels (the `'previous'` curriculum option exists but was not used).
6. **Patches:** random crop from a random training image, accepted only if the GT patch std > 0.05 (contains text; up to 20 tries); random 90° rotation and horizontal flip. **Patch/batch rule: σ ≤ 30 → 128 px × 4; σ ≤ 50 → 192 px × 2; else 256 px × 2.**
7. **Optimisation:** Charbonnier loss (ε = 1e-3); AdamW (lr 2e-5, weight decay 1e-4, betas 0.9/0.999); cosine annealing over 3000 iterations to 1e-7; **3000 iterations**; AMP float16 with GradScaler; gradient-norm clip 1.0; seed 42.
8. **Validation / selection:** every 500 iterations on the 10 validation images (full-image tiled inference); **best validation SSIM** checkpoint saved as `levelN_best.pth`; `levelN_last.pth` holds full resume state.
9. **Inference:** 768 px tiles, 192 px overlap (256 if σ > 50), linear feathering over the outer 96 px, reflect padding to a multiple of 8; output saved as 16-bit PNG.
10. **Evaluation:** PSNR and SSIM (`skimage`, data range 1, whole image) vs registered GT; EasyOCR (`Reader(['en'])`, default models, `detail=1, paragraph=False`), boxes grouped into lines by y, sorted by x, joined by spaces; character accuracy = max(0, 1 − Levenshtein/len(GT)); token accuracy = share of GT tokens found in the OCR tokens (order ignored). CER, WER, exact-match and empty-OCR rates were added post hoc from the saved transcripts.

Per-level training facts:

| Level | Run date | σ est. (px) | Patch × batch | Patches seen | Val PSNR @0 | Val SSIM @0 | Best iter | Val PSNR @best | Val SSIM @best | Loss (first 50 it) | Loss (last 50 it) | Wall time (min) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 2026-11-12 | 5 | 128×128 × 4 | 12000 | 24.71 | 0.5182 | 3000 | 28.17 | 0.6800 | 0.0376 | 0.0279 | 50 |
| 1 | 2026-11-12 | 5 | 128×128 × 4 | 12000 | 22.46 | 0.4579 | 2500 | 28.83 | 0.6277 | 0.0468 | 0.0271 | 45 |
| 2 | 2026-11-12 | 5 | 128×128 × 4 | 12000 | 18.93 | 0.4393 | 3000 | 27.47 | 0.6129 | 0.0661 | 0.0307 | 44 |
| 3 | 2026-11-12 | 12 | 128×128 × 4 | 12000 | 17.03 | 0.4375 | 2000 | 25.76 | 0.6052 | 0.0966 | 0.0355 | 48 |
| 4 | 2026-11-12 | 20 | 128×128 × 4 | 12000 | 15.85 | 0.4249 | 3000 | 23.95 | 0.5939 | 0.1290 | 0.0417 | 43 |
| 5 | 2026-10 (legacy run) | 25 | 128×128 × 4 | 12000 | 15.38 | 0.4225 | 3000 | 22.01 | 0.5832 | 0.1416 | 0.0557 | — |
| 6 | 2026-10-19 | 30 | 128×128 × 4 | 12000 | 15.03 | 0.4172 | 3000 | 20.19 | 0.5684 | 0.1499 | 0.0708 | 45 |
| 7 | 2026-11-11 | 37 | 192×192 × 2 | 6000 | 14.80 | 0.4150 | 2500 | 20.02 | 0.5667 | 0.1401 | 0.0634 | 81 |
| 8 | 2026-11-11 | 45 | 192×192 × 2 | 6000 | 14.43 | 0.4577 | 3000 | 19.48 | 0.5635 | 0.1429 | 0.0748 | 88 |
| 9 | 2026-11-11 | 47 | 192×192 × 2 | 6000 | 14.38 | 0.4583 | 2000 | 16.70 | 0.5286 | 0.1377 | 0.0845 | 85 |

Registration facts:

| Level | σ est. (px) | NCC before | NCC after | NCC after (worst pair) | Affine scale x | Affine scale y | Shift x (px) | Shift y (px) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 5 | 0.897 | 0.949 | 0.938 | 0.9934 | 0.9882 | 7.1 | 13.4 |
| 1 | 5 | 0.950 | 0.970 | 0.966 | 1.0003 | 0.9951 | -0.8 | 10.3 |
| 2 | 5 | 0.906 | 0.960 | 0.953 | 1.0072 | 1.0029 | -7.8 | 6.4 |
| 3 | 12 | 0.864 | 0.948 | 0.941 | 1.0143 | 1.0097 | -14.8 | 3.8 |
| 4 | 20 | 0.824 | 0.915 | 0.905 | 1.0217 | 1.0177 | -23.1 | 0.8 |
| 5 | 25 | 0.792 | 0.892 | 0.878 | 1.0295 | 1.0245 | -32.1 | 0.4 |
| 6 | 30 | 0.758 | 0.867 | 0.849 | 1.0359 | 1.0362 | -37.1 | -8.8 |
| 7 | 37 | 0.722 | 0.836 | 0.805 | 1.0430 | 1.0373 | -48.4 | -2.9 |
| 8 | 45 | 0.669 | 0.804 | 0.755 | 1.0524 | 1.0443 | -55.9 | -4.5 |
| 9 | 47 | 0.626 | 0.780 | 0.730 | 1.0573 | 1.0524 | -68.3 | -8.5 |

---

## 5. Results (test set, 40 images per level)

### 5.1 Headline

| Level | σ est. (px) | PSNR blurred | PSNR fine-tuned | Δ PSNR | SSIM blurred | SSIM fine-tuned | Δ SSIM | Char acc blurred % | Char acc fine-tuned % | Char acc sharp GT % | Δ char acc (pts) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 5 | 26.08 | 28.74 | +2.66 | 0.5236 | 0.6908 | +0.167 | — | — | — | — |
| 1 | 5 | 23.17 | 28.87 | +5.70 | 0.4193 | 0.6265 | +0.207 | 98.4 | 92.6 | 90.8 | -5.7 |
| 2 | 5 | 19.37 | 27.38 | +8.01 | 0.3902 | 0.6151 | +0.225 | 94.2 | 96.1 | 92.2 | +1.8 |
| 3 | 12 | 17.34 | 25.72 | +8.38 | 0.3795 | 0.6056 | +0.226 | 75.5 | 98.3 | 92.5 | +22.7 |
| 4 | 20 | 16.07 | 24.10 | +8.03 | 0.3730 | 0.5949 | +0.222 | 23.5 | 97.4 | 91.3 | +73.9 |
| 5 | 25 | 15.35 | 21.98 | +6.63 | 0.3724 | 0.5824 | +0.210 | 0.0 | 96.9 | 94.1 | +96.9 |
| 6 | 30 | 15.03 | 20.19 | +5.16 | 0.3723 | 0.5693 | +0.197 | 0.0 | 83.2 | 93.4 | +83.2 |
| 7 | 37 | 14.71 | 19.87 | +5.16 | 0.3729 | 0.5665 | +0.194 | 0.0 | 77.0 | 91.6 | +77.0 |
| 8 | 45 | 14.45 | 19.47 | +5.02 | 0.4173 | 0.5629 | +0.146 | 0.0 | 73.5 | 92.6 | +73.5 |
| 9 | 47 | 14.39 | 16.81 | +2.42 | 0.4192 | 0.5310 | +0.112 | 0.0 | 52.8 | 93.2 | +52.8 |

### 5.2 All methods and metrics

Methods: *Blurred input* (CAM02), *Pretrained Restormer* (no fine-tuning), *Fine-tuned Restormer* (this work), *Sharp GT* (EasyOCR on ground truth; image metrics undefined).

| Level | Method | PSNR (dB) | SSIM | Char acc % | Token acc % | CER % | WER % | Exact-match % | Empty OCR % |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | Blurred input | 26.08 | 0.5236 | — | — | — | — | — | — |
| 0 | Pretrained Restormer | 24.98 | 0.5311 | — | — | — | — | — | — |
| 0 | Fine-tuned Restormer | 28.74 | 0.6908 | — | — | — | — | — | — |
| 1 | Blurred input | 23.17 | 0.4193 | 98.35 | 85.79 | 1.65 | 15.88 | 55.0 | 0 |
| 1 | Pretrained Restormer | 22.47 | 0.4552 | 98.11 | 83.71 | 1.89 | 19.62 | 50.0 | 0 |
| 1 | Fine-tuned Restormer | 28.87 | 0.6265 | 92.62 | 60.50 | 7.38 | 56.88 | 20.0 | 0 |
| 1 | Sharp GT (upper bound) | — | — | 90.81 | 57.42 | 9.19 | 64.96 | 15.0 | 0 |
| 2 | Blurred input | 19.37 | 0.3902 | 94.25 | 63.54 | 5.75 | 42.71 | 22.5 | 0 |
| 2 | Pretrained Restormer | 18.84 | 0.4403 | 94.97 | 65.00 | 5.03 | 39.79 | 25.0 | 0 |
| 2 | Fine-tuned Restormer | 27.38 | 0.6151 | 96.07 | 75.00 | 3.93 | 27.50 | 37.5 | 0 |
| 2 | Sharp GT (upper bound) | — | — | 92.16 | 55.62 | 7.84 | 58.54 | 20.0 | 0 |
| 3 | Blurred input | 17.34 | 0.3795 | 75.55 | 24.04 | 24.45 | 88.04 | 2.5 | 0 |
| 3 | Pretrained Restormer | 16.95 | 0.4353 | 75.54 | 23.50 | 24.46 | 93.29 | 2.5 | 0 |
| 3 | Fine-tuned Restormer | 25.72 | 0.6056 | 98.28 | 86.46 | 1.72 | 14.17 | 62.5 | 0 |
| 3 | Sharp GT (upper bound) | — | — | 92.48 | 63.79 | 7.52 | 49.92 | 20.0 | 0 |
| 4 | Blurred input | 16.07 | 0.3730 | 23.48 | 0.50 | 76.52 | 130.17 | 0.0 | 0 |
| 4 | Pretrained Restormer | 15.86 | 0.4241 | 27.34 | 0.50 | 72.66 | 134.96 | 0.0 | 0 |
| 4 | Fine-tuned Restormer | 24.10 | 0.5949 | 97.40 | 82.17 | 2.60 | 17.83 | 47.5 | 0 |
| 4 | Sharp GT (upper bound) | — | — | 91.27 | 55.12 | 8.73 | 65.88 | 15.0 | 0 |
| 5 | Blurred input | 15.35 | 0.3724 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 5 | Pretrained Restormer | 15.29 | 0.4188 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 5 | Fine-tuned Restormer | 21.98 | 0.5824 | 96.94 | 75.00 | 3.06 | 25.83 | 40.0 | 0 |
| 5 | Sharp GT (upper bound) | — | — | 94.11 | 64.71 | 5.89 | 51.54 | 25.0 | 0 |
| 6 | Blurred input | 15.03 | 0.3723 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 6 | Pretrained Restormer | 15.05 | 0.4174 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 6 | Fine-tuned Restormer | 20.19 | 0.5693 | 83.22 | 32.00 | 16.78 | 78.83 | 0.0 | 0 |
| 6 | Sharp GT (upper bound) | — | — | 93.41 | 69.12 | 6.59 | 38.67 | 20.0 | 0 |
| 7 | Blurred input | 14.71 | 0.3729 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 7 | Pretrained Restormer | 14.76 | 0.4171 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 7 | Fine-tuned Restormer | 19.87 | 0.5665 | 77.04 | 22.29 | 22.96 | 98.12 | 0.0 | 0 |
| 7 | Sharp GT (upper bound) | — | — | 91.63 | 61.67 | 8.37 | 61.25 | 20.0 | 0 |
| 8 | Blurred input | 14.45 | 0.4173 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 8 | Pretrained Restormer | 14.44 | 0.4579 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 8 | Fine-tuned Restormer | 19.47 | 0.5629 | 73.47 | 23.92 | 26.53 | 98.38 | 0.0 | 0 |
| 8 | Sharp GT (upper bound) | — | — | 92.62 | 63.83 | 7.38 | 56.50 | 17.5 | 0 |
| 9 | Blurred input | 14.39 | 0.4192 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 9 | Pretrained Restormer | 14.38 | 0.4595 | 0.00 | 0.00 | 100.00 | 100.00 | 0.0 | 100 |
| 9 | Fine-tuned Restormer | 16.81 | 0.5310 | 52.82 | 10.87 | 47.18 | 133.12 | 0.0 | 0 |
| 9 | Sharp GT (upper bound) | — | — | 93.24 | 66.79 | 6.76 | 50.17 | 32.5 | 0 |

### 5.3 Gains and per-image consistency

| Level | Δ PSNR vs blurred | Δ SSIM vs blurred | Δ char acc vs blurred (pts) | Δ token acc vs blurred (pts) | Δ PSNR vs pretrained | Δ SSIM vs pretrained | Δ char acc vs pretrained (pts) | Images with PSNR gain | Images with SSIM gain | Char acc win / tie / loss vs blurred | Fine-tuned − sharp-GT char acc (pts) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | +2.66 | +0.1672 | — | — | +3.75 | +0.1596 | — | 40/40 | 40/40 | — | — |
| 1 | +5.70 | +0.2072 | -5.7 | -25.3 | +6.40 | +0.1713 | -5.5 | 40/40 | 40/40 | 1 / 11 / 28 | +1.8 |
| 2 | +8.01 | +0.2249 | +1.8 | +11.5 | +8.54 | +0.1748 | +1.1 | 40/40 | 40/40 | 18 / 17 / 5 | +3.9 |
| 3 | +8.38 | +0.2261 | +22.7 | +62.4 | +8.77 | +0.1703 | +22.7 | 40/40 | 40/40 | 38 / 2 / 0 | +5.8 |
| 4 | +8.03 | +0.2220 | +73.9 | +81.7 | +8.24 | +0.1709 | +70.1 | 40/40 | 40/40 | 40 / 0 / 0 | +6.1 |
| 5 | +6.63 | +0.2099 | +96.9 | +75.0 | +6.69 | +0.1636 | +96.9 | 40/40 | 40/40 | 40 / 0 / 0 | +2.8 |
| 6 | +5.16 | +0.1970 | +83.2 | +32.0 | +5.14 | +0.1519 | +83.2 | 40/40 | 40/40 | 40 / 0 / 0 | -10.2 |
| 7 | +5.16 | +0.1936 | +77.0 | +22.3 | +5.11 | +0.1494 | +77.0 | 40/40 | 40/40 | 40 / 0 / 0 | -14.6 |
| 8 | +5.02 | +0.1456 | +73.5 | +23.9 | +5.03 | +0.1050 | +73.5 | 40/40 | 40/40 | 40 / 0 / 0 | -19.1 |
| 9 | +2.42 | +0.1118 | +52.8 | +10.9 | +2.43 | +0.0715 | +52.8 | 40/40 | 40/40 | 40 / 0 / 0 | -40.4 |

### 5.4 By font (fine-tuned and baselines)

| Level | Method | Font | PSNR (dB) | SSIM | Char acc % | Token acc % | CER % | WER % |
|--:|:--|:--|--:|--:|--:|--:|--:|--:|
| 0 | Blurred input | timesR | 27.11 | 0.5663 | — | — | — | — |
| 0 | Blurred input | verdanaRef | 25.05 | 0.4809 | — | — | — | — |
| 0 | Pretrained Restormer | timesR | 25.58 | 0.5606 | — | — | — | — |
| 0 | Pretrained Restormer | verdanaRef | 24.39 | 0.5017 | — | — | — | — |
| 0 | Fine-tuned Restormer | timesR | 30.41 | 0.7180 | — | — | — | — |
| 0 | Fine-tuned Restormer | verdanaRef | 27.06 | 0.6635 | — | — | — | — |
| 1 | Blurred input | timesR | 23.12 | 0.4251 | 98.44 | 86.17 | 1.56 | 17.17 |
| 1 | Blurred input | verdanaRef | 23.22 | 0.4136 | 98.27 | 85.42 | 1.73 | 14.58 |
| 1 | Pretrained Restormer | timesR | 22.39 | 0.4600 | 97.96 | 82.00 | 2.04 | 24.67 |
| 1 | Pretrained Restormer | verdanaRef | 22.54 | 0.4504 | 98.27 | 85.42 | 1.73 | 14.58 |
| 1 | Fine-tuned Restormer | timesR | 29.54 | 0.6327 | 89.49 | 51.75 | 10.51 | 77.83 |
| 1 | Fine-tuned Restormer | verdanaRef | 28.20 | 0.6202 | 95.75 | 69.25 | 4.25 | 35.92 |
| 1 | Sharp GT (upper bound) | timesR | — | — | 87.27 | 45.58 | 12.73 | 88.58 |
| 1 | Sharp GT (upper bound) | verdanaRef | — | — | 94.35 | 69.25 | 5.65 | 41.33 |
| 2 | Blurred input | timesR | 19.21 | 0.3943 | 90.53 | 45.83 | 9.47 | 66.67 |
| 2 | Blurred input | verdanaRef | 19.52 | 0.3861 | 97.97 | 81.25 | 2.03 | 18.75 |
| 2 | Pretrained Restormer | timesR | 18.73 | 0.4450 | 91.81 | 47.50 | 8.19 | 62.08 |
| 2 | Pretrained Restormer | verdanaRef | 18.96 | 0.4356 | 98.12 | 82.50 | 1.88 | 17.50 |
| 2 | Fine-tuned Restormer | timesR | 27.90 | 0.6214 | 93.85 | 65.83 | 6.15 | 39.17 |
| 2 | Fine-tuned Restormer | verdanaRef | 26.86 | 0.6089 | 98.28 | 84.17 | 1.72 | 15.83 |
| 2 | Sharp GT (upper bound) | timesR | — | — | 88.69 | 42.50 | 11.31 | 84.17 |
| 2 | Sharp GT (upper bound) | verdanaRef | — | — | 95.62 | 68.75 | 4.38 | 32.92 |
| 3 | Blurred input | timesR | 17.39 | 0.3831 | 60.18 | 3.00 | 39.82 | 119.50 |
| 3 | Blurred input | verdanaRef | 17.28 | 0.3759 | 90.92 | 45.08 | 9.08 | 56.58 |
| 3 | Pretrained Restormer | timesR | 17.05 | 0.4382 | 60.64 | 6.50 | 39.36 | 125.42 |
| 3 | Pretrained Restormer | verdanaRef | 16.86 | 0.4324 | 90.44 | 40.50 | 9.56 | 61.17 |
| 3 | Fine-tuned Restormer | timesR | 25.71 | 0.6092 | 99.06 | 92.75 | 0.94 | 8.50 |
| 3 | Fine-tuned Restormer | verdanaRef | 25.73 | 0.6020 | 97.49 | 80.17 | 2.51 | 19.83 |
| 3 | Sharp GT (upper bound) | timesR | — | — | 89.80 | 55.58 | 10.20 | 67.92 |
| 3 | Sharp GT (upper bound) | verdanaRef | — | — | 95.15 | 72.00 | 4.85 | 31.92 |
| 4 | Blurred input | timesR | 16.30 | 0.3778 | 14.59 | 0.00 | 85.41 | 117.67 |
| 4 | Blurred input | verdanaRef | 15.83 | 0.3682 | 32.36 | 1.00 | 67.64 | 142.67 |
| 4 | Pretrained Restormer | timesR | 16.11 | 0.4286 | 18.07 | 0.00 | 81.93 | 125.08 |
| 4 | Pretrained Restormer | verdanaRef | 15.61 | 0.4196 | 36.60 | 1.00 | 63.40 | 144.83 |
| 4 | Fine-tuned Restormer | timesR | 23.92 | 0.5962 | 96.86 | 79.67 | 3.14 | 20.33 |
| 4 | Fine-tuned Restormer | verdanaRef | 24.28 | 0.5937 | 97.93 | 84.67 | 2.07 | 15.33 |
| 4 | Sharp GT (upper bound) | timesR | — | — | 88.54 | 44.42 | 11.46 | 86.75 |
| 4 | Sharp GT (upper bound) | verdanaRef | — | — | 93.99 | 65.83 | 6.01 | 45.00 |
| 5 | Blurred input | timesR | 15.58 | 0.3780 | 0.00 | 0.00 | 100.00 | 100.00 |
| 5 | Blurred input | verdanaRef | 15.12 | 0.3669 | 0.00 | 0.00 | 100.00 | 100.00 |
| 5 | Pretrained Restormer | timesR | 15.53 | 0.4243 | 0.00 | 0.00 | 100.00 | 100.00 |
| 5 | Pretrained Restormer | verdanaRef | 15.04 | 0.4133 | 0.00 | 0.00 | 100.00 | 100.00 |
| 5 | Fine-tuned Restormer | timesR | 21.77 | 0.5839 | 96.71 | 73.25 | 3.29 | 26.75 |
| 5 | Fine-tuned Restormer | verdanaRef | 22.18 | 0.5809 | 97.16 | 76.75 | 2.84 | 24.92 |
| 5 | Sharp GT (upper bound) | timesR | — | — | 92.15 | 58.08 | 7.85 | 66.92 |
| 5 | Sharp GT (upper bound) | verdanaRef | — | — | 96.07 | 71.33 | 3.93 | 36.17 |
| 6 | Blurred input | timesR | 15.32 | 0.3766 | 0.00 | 0.00 | 100.00 | 100.00 |
| 6 | Blurred input | verdanaRef | 14.75 | 0.3680 | 0.00 | 0.00 | 100.00 | 100.00 |
| 6 | Pretrained Restormer | timesR | 15.35 | 0.4220 | 0.00 | 0.00 | 100.00 | 100.00 |
| 6 | Pretrained Restormer | verdanaRef | 14.75 | 0.4127 | 0.00 | 0.00 | 100.00 | 100.00 |
| 6 | Fine-tuned Restormer | timesR | 19.83 | 0.5683 | 78.52 | 23.92 | 21.48 | 92.75 |
| 6 | Fine-tuned Restormer | verdanaRef | 20.55 | 0.5703 | 87.91 | 40.08 | 12.09 | 64.92 |
| 6 | Sharp GT (upper bound) | timesR | — | — | 91.69 | 66.42 | 8.31 | 44.58 |
| 6 | Sharp GT (upper bound) | verdanaRef | — | — | 95.12 | 71.83 | 4.88 | 32.75 |
| 7 | Blurred input | timesR | 14.93 | 0.3751 | 0.00 | 0.00 | 100.00 | 100.00 |
| 7 | Blurred input | verdanaRef | 14.50 | 0.3707 | 0.00 | 0.00 | 100.00 | 100.00 |
| 7 | Pretrained Restormer | timesR | 14.99 | 0.4192 | 0.00 | 0.00 | 100.00 | 100.00 |
| 7 | Pretrained Restormer | verdanaRef | 14.54 | 0.4149 | 0.00 | 0.00 | 100.00 | 100.00 |
| 7 | Fine-tuned Restormer | timesR | 19.33 | 0.5631 | 66.34 | 5.42 | 33.66 | 130.00 |
| 7 | Fine-tuned Restormer | verdanaRef | 20.42 | 0.5699 | 87.74 | 39.17 | 12.26 | 66.25 |
| 7 | Sharp GT (upper bound) | timesR | — | — | 87.95 | 53.33 | 12.05 | 86.25 |
| 7 | Sharp GT (upper bound) | verdanaRef | — | — | 95.30 | 70.00 | 4.70 | 36.25 |
| 8 | Blurred input | timesR | 14.66 | 0.4199 | 0.00 | 0.00 | 100.00 | 100.00 |
| 8 | Blurred input | verdanaRef | 14.24 | 0.4147 | 0.00 | 0.00 | 100.00 | 100.00 |
| 8 | Pretrained Restormer | timesR | 14.65 | 0.4604 | 0.00 | 0.00 | 100.00 | 100.00 |
| 8 | Pretrained Restormer | verdanaRef | 14.23 | 0.4554 | 0.00 | 0.00 | 100.00 | 100.00 |
| 8 | Fine-tuned Restormer | timesR | 18.75 | 0.5589 | 57.47 | 8.50 | 42.53 | 129.42 |
| 8 | Fine-tuned Restormer | verdanaRef | 20.19 | 0.5669 | 89.48 | 39.33 | 10.52 | 67.33 |
| 8 | Sharp GT (upper bound) | timesR | — | — | 91.37 | 60.17 | 8.63 | 70.83 |
| 8 | Sharp GT (upper bound) | verdanaRef | — | — | 93.88 | 67.50 | 6.12 | 42.17 |
| 9 | Blurred input | timesR | 14.60 | 0.4216 | 0.00 | 0.00 | 100.00 | 100.00 |
| 9 | Blurred input | verdanaRef | 14.17 | 0.4168 | 0.00 | 0.00 | 100.00 | 100.00 |
| 9 | Pretrained Restormer | timesR | 14.59 | 0.4619 | 0.00 | 0.00 | 100.00 | 100.00 |
| 9 | Pretrained Restormer | verdanaRef | 14.17 | 0.4571 | 0.00 | 0.00 | 100.00 | 100.00 |
| 9 | Fine-tuned Restormer | timesR | 16.55 | 0.5299 | 34.33 | 3.25 | 65.67 | 160.83 |
| 9 | Fine-tuned Restormer | verdanaRef | 17.06 | 0.5320 | 71.30 | 18.50 | 28.70 | 105.42 |
| 9 | Sharp GT (upper bound) | timesR | — | — | 90.26 | 54.08 | 9.74 | 74.67 |
| 9 | Sharp GT (upper bound) | verdanaRef | — | — | 96.22 | 79.50 | 3.78 | 25.67 |

---

## 6. Level-by-level findings

### Level 0

- **What this level is.** Step 0 is the mildest setting in the dataset: the "blurred" camera is only slightly out of focus, so the input already scores 26.08 dB / 0.524 SSIM against the ground truth.
- **Restoration quality.** Fine-tuning still helps: 28.74 dB / 0.691 SSIM (+2.66 dB, +0.167 SSIM), and every one of the 40 test images improves on both metrics. The un-tuned pretrained model *lowers* PSNR (24.98 dB), which is the same domain-mismatch pattern seen at every level.
- **Font gap.** Times outputs reach 30.41 dB while Verdana reaches 27.06 dB. The same gap is present in the blurred inputs (27.11 vs 25.05 dB), so it comes from the data (the Verdana pairs at this step differ more from their ground truth), not from the model.
- **Gap in the exported results.** The Level 0 folder has a best/last checkpoint, training log, split, registration files and all 80 restored PNGs, but **no `level0_summary.csv`, `level0_per_image.csv`, `ocr_outputs.csv` or `manifest.json`**. The OCR/summary cells were therefore not run (or not saved) for this level. The PSNR/SSIM values here were recomputed for this report from the saved restored images and the dataset, using the notebook's exact preprocessing; the same recomputation reproduces the notebook's own CSV values for Levels 1–9 to within 0.003 dB. **OCR accuracy for Level 0 is not available.**

### Level 1

- **Image quality improves strongly.** +5.70 dB PSNR and +0.207 SSIM over the blurred input; all 40 test images improve.
- **OCR accuracy goes down, not up.** EasyOCR already reads the Level 1 blurred input at 98.35% character accuracy. After restoration it reads 92.62% (−5.7 points); 28 of 40 images get worse, 11 tie and 1 improves.
- **Why.** The sharp ground truth itself only scores 90.81%. On the sharp, high-contrast renderings EasyOCR repeatedly drops wide glyphs (for example `pApReWCcGs` is read as `pApRe CcGs` on both the ground truth and the fine-tuned output, but correctly on the blurred input). The fine-tuned model makes the image look like the ground truth, so it inherits the recogniser's behaviour on ground-truth-like images. At this level the blur is too mild to hurt recognition, so deblurring has nothing to gain on the OCR metric.
- **Practical reading.** For Level 1 (and by extension Level 0) the restoration is a visual-quality improvement only; a recognition-driven pipeline should pass these images through unchanged.

### Level 2

- **First level where restoration helps OCR.** Character accuracy rises from 94.25% to 96.07% and token accuracy from 63.5% to 75.0%; 18 images improve, 17 tie, 5 get worse.
- **PSNR gain jumps to +8.01 dB** (19.37 → 27.38 dB), SSIM +0.225.
- **Blur estimate caveat.** The estimated σ is 5 px for Levels 0, 1 and 2 alike. 5 px is the smallest value in the search grid, so for these three levels the estimate is a floor, not a measurement; the PSF images in REPORT.md §3 show the blur does grow from step 0 to step 2.

### Level 3

- **Best OCR result of the whole study.** Fine-tuned character accuracy is 98.28% and token accuracy 86.5%, with 62.5% of the test images transcribed with zero errors. The blurred input reads 75.55% / 24.0%.
- **Largest PSNR gain.** +8.38 dB (17.34 → 25.72 dB). 38 of 40 images improve on character accuracy, 2 tie, none get worse.
- **Above the "upper bound".** The fine-tuned output is read more accurately than the sharp ground truth (98.28% vs 92.48%). The slightly smoother, noise-free strokes produced by the network suit EasyOCR better than the grainy camera image (see REPORT.md §9.2).

### Level 4

- **Blurred input is now unreadable in practice.** 23.5% character accuracy and 0.5% token accuracy on the input; the pretrained model does not change that (27.3% / 0.5%).
- **Restoration recovers it almost completely.** 97.40% character accuracy and 82.2% token accuracy; all 40 images improve. PSNR +8.03 dB, SSIM +0.222.
- **Registration starts to matter.** The alignment score after warping drops to 0.915 (from 0.948 at Level 3). The blur is now wide enough (σ ≈ 20 px) that matching the sharp image to the blurred one is less precise.

### Level 5

- **The headline level.** EasyOCR detects **no text at all** on any of the 40 blurred inputs or any of the 40 pretrained-Restormer outputs (0% accuracy, 100% empty outputs). After fine-tuning it reads 96.94% of characters and 75.0% of tokens, and 40% of images are transcribed perfectly.
- **Image metrics.** 15.35 → 21.98 dB (+6.63 dB), SSIM 0.372 → 0.582 (+0.210).
- **Legacy run.** Level 5 was trained first, with the earlier version of the notebook, before the Drive folder was reorganised. Its results sit one folder deeper (`level_05/results/level5_results/`), its `manifest.json` has no training configuration, and there is no `train_log.txt`. Blur σ ≈ 25 px comes from the notebook comment and from the combined-model preprocessing log; patch 128 × batch 4 follows from the notebook's rule for σ ≤ 30 and its comment that Level 5 used 128 × 4.

### Level 6

- **Clear drop from Level 5.** Character accuracy 83.22% (from 96.94% at Level 5) and token accuracy 32.0% (from 75.0%). No image is transcribed perfectly any more. The input is still completely unreadable (0%).
- **Image metrics.** 15.03 → 20.19 dB (+5.16 dB), SSIM +0.197.
- **A probable contributor: patch size.** σ ≈ 30 px sits exactly on the boundary of the notebook's rule, so Level 6 was trained with 128 px patches. A 128 px patch is only about 4σ wide, i.e. smaller than the blur's footprint (roughly ±2–3σ), so much of what the network needs to undo the blur lies outside the patch. Levels 7–9 use 192 px patches. Re-running Level 6 with `PATCH_OVERRIDE = (192, 2)` is a cheap experiment to check this.
- **Fonts diverge.** Times 78.5% vs Verdana 87.9% character accuracy; thin serif strokes are lost first.

### Level 7

- **Still usable at character level, weak at word level.** 77.04% character accuracy, 22.3% token accuracy; PSNR +5.16 dB (14.71 → 19.87 dB), SSIM +0.194.
- **Training setup changes here.** σ ≈ 37 px triggers 192 px patches with batch 2, so the model sees 6,000 patches instead of 12,000 and each iteration takes about twice as long (81 min total vs ~45 min).
- **Strong font effect.** Times 66.3% vs Verdana 87.7% character accuracy.

### Level 8

- **Similar to Level 7 on images, slightly lower on OCR.** 19.47 dB / 0.563 SSIM; 73.47% character accuracy and 23.9% token accuracy.
- **Font effect widens further.** Times 57.5% vs Verdana 89.5%. Verdana at Level 8 is read about as well as at Level 6–7; almost all of the loss is on the serif font.
- **Input SSIM is higher than at Level 7 (0.417 vs 0.373) although the blur is stronger.** The same jump appears for the pretrained model. SSIM against a noisy ground truth is dominated by background statistics rather than text here, so SSIM values should be compared within a level, not across levels (see REPORT.md §10).

### Level 9

- **The method is near its limit.** PSNR gain falls to +2.42 dB (14.39 → 16.81 dB), SSIM +0.112, character accuracy 52.82% and token accuracy 10.9%. Every image still improves over the (completely unreadable) input.
- **Training was unstable.** Validation PSNR moved 15.48 → 15.44 → 15.12 → 16.70 → 15.88 → 16.12 dB across checkpoints and the best checkpoint (by SSIM) is iteration 2000, not the last one. The loss is still falling at iteration 3000 (0.0845), so the model is under-trained at this level.
- **Alignment is the weakest of all levels.** Mean alignment score 0.780, worst pair 0.730. With σ ≈ 47 px the registration itself is uncertain, which caps both the training signal and the measured PSNR.
- **Outputs show invented strokes.** In the close-up figure the Times outputs contain blobs and broken strokes that are not in the ground truth; Verdana remains partly legible (71.3% vs 34.3% for Times).

---

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
ultimate_model/{cache,logs,results,checkpoints}
```

**This report folder (`Restormer_Report_Levels0-9/`)**

```
REPORT.md / .docx / .pdf / .html      full report
PROJECT_CONTEXT.md                    this file
tables/                               COMPARISON_TABLES.md + CSVs (summary, per-image, by-font, gains, training/registration, OCR transcripts)
comparisons/level_NN_comparison.jpg   blurred | fine-tuned | ground truth, 4 test images, with metrics and OCR text
comparisons/level_NN_closeup.jpg      native-resolution centre crops
images/level_NN/                      individual panels: L<N>_<id>_{blurred,pretrained,finetuned,groundtruth}.jpg
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
