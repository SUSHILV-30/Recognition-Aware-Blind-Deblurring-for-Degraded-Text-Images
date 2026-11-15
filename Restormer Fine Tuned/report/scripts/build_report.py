from build_common import *
from level_sections import level_section
T=P['total']; FZ=P['frozen']; TRN=P['trainable']
R=[]
R.append(f"""# Restormer Fine-Tuning for Defocus-Blurred Text — Complete Report, Blur Levels 0–9

- **Project:** Scene Text Recognition System (Final Year Project, Phase 2) — recognition-aware blind deblurring of degraded text images
- **Report date:** 12 November 2026
- **Source material:** the Drive export `FinalYearProject-Phase2` (11 zip parts), the ten dataset archives `HDC2021_step0 … step9` (plus step 10), and the training notebook `FinalYearProject_Phase2_Level1.ipynb`
- **Companion files in this folder:** `PROJECT_CONTEXT.md` (compact context for a new reader or LLM), `tables/` (all numbers as CSV and Markdown), `comparisons/` (image grids), `images/` (individual full-resolution panels), `figures/` (diagrams and charts)

---

## 1. Summary

One Restormer model per blur level was fine-tuned on 150 image pairs of the HDC 2021 defocus dataset and tested on 40 held-out pairs. The same notebook, hyper-parameters and image split were used at every level; only the blur level changed.

{headline_table()}
*Level 0 has no OCR results in the export; its PSNR/SSIM were recomputed for this report (§8.1). σ = 5 px is the minimum of the search grid. "Sharp GT" is EasyOCR run on the registered ground-truth image.*

![Trends across levels](figures/trends_all_levels.png)

**Main findings**

1. **Fine-tuning improves image quality at every level.** PSNR gains over the blurred input range from +2.42 dB (Level 9) to +8.38 dB (Level 3); SSIM gains from +0.112 to +0.226. All 400 test images (40 × 10 levels) improve on both PSNR and SSIM.
2. **The pretrained Restormer does not help on this data.** Without fine-tuning its PSNR is never more than 0.05 dB above the blurred input and is up to 1.1 dB below it (Levels 0–5), and its OCR accuracy stays within 4 points of the blurred input. The gain comes from fine-tuning, not from the architecture alone.
3. **OCR benefit depends on the level.** At Level 1 restoration lowers character accuracy (98.4% → 92.6%) because the input is already readable. From Level 2 upward it helps, and from Level 5 upward EasyOCR finds no text at all in the blurred input while the restored images are read at 96.9% (Level 5), 83.2% (6), 77.0% (7), 73.5% (8) and 52.8% (9) character accuracy.
4. **Levels 2–5 are read better than the sharp ground truth.** EasyOCR scores only 90.8–94.1% on the ground-truth images themselves, so the "upper bound" row is a reference point, not a ceiling.
5. **Serif text fails first.** From Level 6 the Times outputs lose accuracy much faster than Verdana (Level 9: 34.3% vs 71.3%).
6. **Level 9 is at the limit of this setup.** Validation metrics oscillate, the loss has not converged after 3000 iterations, registration quality is the lowest, and outputs contain strokes that are not in the ground truth.

**Status of the export.** Levels 1–9: complete (checkpoints, logs, metrics, OCR outputs, restored images). Level 0: trained and restored, but OCR/summary files are missing. Level 10: dataset present, not trained. Combined "ultimate" model (Levels 5–10): preprocessing done and training started on 20 October 2026; the export contains only a last checkpoint and no evaluation (§11).

---

## 2. What the export contains

| Item | Location in the Drive export | Notes |
|:--|:--|:--|
| Pretrained weights | `pretrained/single_image_defocus_deblurring.pth` (104.7 MB) | Official Restormer single-image defocus-deblurring weights; starting point for every level |
| Per-level folders | `level_00` … `level_10` | `checkpoints/`, `results/`, `logs/`, `manifest.json` |
| Best / last checkpoints | `level_XX/checkpoints/levelN_best.pth`, `levelN_last.pth` | Present for Levels 0–9 |
| Metrics | `results/levelN_summary.csv`, `levelN_per_image.csv`, `levelN_by_font.csv` | Levels 1–9 (Level 5 under `results/level5_results/`) |
| OCR transcripts | `results/ocr_outputs.csv` | 160 rows per level (4 methods × 40 images), Levels 1–9 |
| Restored images | `results/restored/pretrained/*.png`, `results/restored/finetuned/*.png` | 16-bit PNG, 40 + 40 per level, Levels 0–9 |
| Registration | `results/registration_affine.npy`, `results/pairing_checks.csv` | One affine per level; alignment score for all 200 pairs |
| Split | `results/split.json` | Identical image IDs at every level |
| Training history | `logs/training_history.json`, `logs/train_log.txt` | Level 5 has history only (legacy run) |
| Cross-level summary | `all_levels_summary.csv`, `all_levels_trend.png` | Levels 1–9 |
| Combined model | `ultimate_model/` | Registration cache, splits, log, `ultimate_last.pth` (418.8 MB) |
| Datasets | `HDC2021_step0.zip` … `HDC2021_step_10.zip` (≈2.5 GB each) | Downloaded separately, same folder |

---

## 3. Dataset: HDC 2021 defocus steps

The data come from the Helsinki Deblur Challenge 2021 (HDC 2021). Text targets shown on an e-ink display were photographed simultaneously by two cameras through a beam splitter: **CAM01 stays in focus (ground truth)** and **CAM02 is progressively defocused (input)**. Each "focus step" is one fixed amount of defocus; a higher step means more blur.

Facts verified directly from the archives used here:

- Each step archive holds `stepN/Times/CAM01`, `Times/CAM02`, `Verdana/CAM01`, `Verdana/CAM02` with **100 samples per font per camera** → 200 sharp/blurred pairs per level.
- Files are named `focusStep_N_<timesR|verdanaRef>_size_30_sample_XXXX.tif`; each has a `.txt` file with the true text.
- Images are single-channel TIFFs of **2360 × 1460 px**. After registration and a 48 px border trim the working size is **2264 × 1364 px**.
- Each image shows three lines of random upper/lower-case letters (about 30 characters). The text is random, so a language model cannot guess it; the recogniser has to read every glyph.
- Each folder also ships point-target and line-target images (`focusStep_N_PSF.tif`, `_LSF_X.tif`, `_LSF_Y.tif`). The point targets below show how the blur disc grows with the step.

![PSF by level](figures/psf_by_level.png)

The blur is real optical defocus (a disc-like kernel plus sensor noise and a small change of magnification), not a synthetic Gaussian. The two cameras are also not pixel-aligned, which is why registration (§5.4) is needed before any pixel-wise loss or metric can be used.

---

## 4. The Restormer architecture

Restormer (Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration", CVPR 2022) is an encoder–decoder Transformer built for image restoration at full resolution. This project uses the authors' **single-image defocus-deblurring configuration** unchanged, loaded from the official repository file `basicsr/models/archs/restormer_arch.py`:

```python
Restormer(inp_channels=3, out_channels=3, dim=48,
          num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
          heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
          bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
```

![Restormer architecture](figures/restormer_architecture.png)

### 4.1 Overall data flow

1. **Input.** A blurred image `I` of size H × W × 3. The text images are grayscale, so the single channel is copied three times to match the pretrained weights.
2. **Overlapping patch embedding.** One 3 × 3 convolution maps 3 → 48 channels at full resolution. Unlike ViT there is no 16 × 16 patch cutting; every pixel keeps its own feature vector, which preserves thin strokes.
3. **Encoder, three levels.** Level 1: 4 Transformer blocks at 48 channels, full resolution. Level 2: 6 blocks at 96 channels, half resolution. Level 3: 6 blocks at 192 channels, quarter resolution. Between levels a 3 × 3 convolution halves the channel count and `PixelUnshuffle(2)` folds each 2 × 2 neighbourhood into channels, so resolution halves and channels double without discarding any values.
4. **Latent (bottleneck).** 8 Transformer blocks at 384 channels and 1/8 resolution. This is where the receptive field is largest and where 55% of all parameters sit.
5. **Decoder, three levels.** Each step upsamples with a 3 × 3 convolution followed by `PixelShuffle(2)`, concatenates the encoder features of the same resolution (skip connection), and — at Levels 3 and 2 — reduces the doubled channel count with a 1 × 1 convolution before 6 Transformer blocks. At Level 1 the concatenated 96 channels are kept (no reduction) and passed through 4 blocks.
6. **Refinement.** 4 more Transformer blocks at full resolution and 96 channels sharpen fine detail.
7. **Output.** A 3 × 3 convolution maps 96 → 3 channels to give a residual image `R`; the result is `Î = I + R`.

*In plain words:* the encoder looks at the page from further and further away to understand how wide the blur is; the bottleneck decides what the text must have been; the decoder zooms back in, each time re-reading the encoder's notes for that zoom level so that fine positions are not lost; and the network only has to output the *correction* to add to the blurred picture, not the whole picture.

### 4.2 The Transformer block

Every one of the 44 blocks has the same structure:

```
X'    = X  + MDTA(LayerNorm(X))
X_out = X' + GDFN(LayerNorm(X'))
```

![Transformer block detail](figures/transformer_block_detail.png)

**LayerNorm (`WithBias`).** For each pixel, the feature vector across channels is normalised to zero mean and unit variance and then scaled and shifted by learned per-channel weight and bias. It is applied before each sub-layer (pre-norm), which keeps training stable in deep stacks.

**MDTA — Multi-Dconv head Transposed Attention.**

1. A 1 × 1 convolution expands C → 3C channels; a 3 × 3 depth-wise convolution then mixes each channel with its 3 × 3 neighbourhood. The result is split into query Q, key K and value V (C channels each).
2. Q, K and V are reshaped to `heads × (C/heads) × (H·W)`: each row is one channel laid out as a long vector over all pixels.
3. Q and K are L2-normalised along the pixel axis.
4. The attention map is `A = softmax(τ · Q Kᵀ)`, a `(C/heads) × (C/heads)` matrix per head, where τ is a learned temperature (one scalar per head).
5. The output is `A · V`, reshaped back to C × H × W and passed through a 1 × 1 convolution.

Standard self-attention relates every pixel to every other pixel, which needs an (H·W) × (H·W) map — impossible for megapixel images. MDTA "transposes" the problem: it relates every **channel** to every other channel. The map is tiny (48 × 48 per head in the encoder, latent and decoder levels 3 and 2; 96 × 96 in decoder level 1 and refinement), cost grows linearly with the number of pixels, and each entry of `Q Kᵀ` is still a sum over the whole image, so the block has global context. Local structure is supplied by the depth-wise convolution.

*In plain words:* instead of asking "which other pixels should this pixel look at?", the block asks "which feature detectors fire together across the whole picture?" and re-mixes the feature maps accordingly. For blur this is useful because blur is the same kind of damage everywhere on the page.

**GDFN — Gated-Dconv Feed-Forward Network.**

1. A 1 × 1 convolution expands C → 2·⌊2.66·C⌋ channels.
2. A 3 × 3 depth-wise convolution adds spatial context.
3. The tensor is split into two halves `x₁`, `x₂`; the output is `GELU(x₁) ⊙ x₂` (element-wise product).
4. A 1 × 1 convolution projects ⌊2.66·C⌋ → C.

The product acts as a gate: one half decides how much of the other half passes at each position. Flat paper regions can be suppressed while stroke edges are passed on. Hidden widths in this model are 127, 255, 510 and 1021 for C = 48, 96, 192 and 384.

### 4.3 Design details that matter for this project

- **No bias in any convolution** (`bias=False`); the only biases are in LayerNorm.
- **Pixel-(un)shuffle sampling** changes resolution by rearranging values instead of pooling or interpolating, so no stroke information is averaged away on the way down and no checkerboard-prone transposed convolutions are used on the way up.
- **Heads double with depth** (1, 2, 4, 8) so each head works on 48 channels in the encoder, latent and deeper decoder levels (96 in the single-head full-resolution decoder and refinement stages).
- **Global residual** means an untrained output layer returns the input unchanged; fine-tuning starts from "do nothing harmful".
- **Fully convolutional.** There is no positional embedding and no fixed input size, so the model trained on 128–192 px patches can be run on 768 px tiles at test time. Input height and width must be multiples of 8 (three halvings).
- **Receptive field.** Convolutions alone see a limited neighbourhood, but channel attention pools over the entire patch or tile. This is one reason patch size matters during training: with a patch smaller than the blur footprint, the global statistics the attention sees in training differ from those in a 768 px test tile.

### 4.4 Parameter budget and what is frozen

{param_table()}
The counts were derived analytically from the layer definitions and reproduce the notebook's printed values exactly: total {T/1e6:.2f} M, frozen {FZ/1e6:.2f} M, trainable {TRN/1e6:.2f} M.

**Frozen:** `patch_embed`, `encoder_level1`, `down1_2`, `encoder_level2`, `down2_3`, `encoder_level3`, `down3_4` — the whole encoder path, {100*FZ/T:.1f}% of the parameters.
**Trained:** latent, all three decoder levels, the up-sampling and channel-reduction layers, refinement and output — {100*TRN/T:.1f}% of the parameters.

The reasoning behind this split: the encoder's early layers extract generic edges and blur cues that transfer from natural images, while the deeper layers decide *what the sharp content should be*, which is domain-specific (printed glyphs rather than natural scenes). Freezing the encoder also saves memory and computation on a T4 (no gradients are stored for it) and removes one source of over-fitting with only 150 training images. Note that although the encoder holds most of the *blocks at high resolution*, it holds few parameters; in parameter terms most of the network is still trained.

---

## 5. Training method (as implemented in the notebook)

The notebook is written so that only `LEVEL` changes between runs. Everything below was read from the code and confirmed against the saved logs and manifests.

![Pipeline](figures/pipeline_overview.png)

### 5.1 Environment and reproducibility

| Item | Value |
|:--|:--|
| Platform | Google Colab, Tesla T4 (15.6 GB), PyTorch 2.14.1+cu130 (as printed by the notebook) |
| Libraries | `einops`, `tifffile`, `scikit-image`, `easyocr`, `rapidfuzz`, OpenCV, pandas, matplotlib |
| Random seed | 42 for Python, NumPy and PyTorch |
| Model code | `git clone https://github.com/swz30/Restormer`; only `restormer_arch.py` is imported (avoids `basicsr` install problems) |
| Storage layout | `FinalYearProject-Phase2/level_XX/{{dataset,checkpoints,results,logs}}` + `manifest.json`; dataset zip copied to local disk and unzipped for speed |
| Resume | `RESUME=True` reloads `levelN_last.pth` (model, optimiser, scheduler, AMP scaler, iteration, best SSIM, history) |
| Initialisation switch | `INIT_FROM='pretrained'` (used for every level) or `'previous'` (start from the previous level's best checkpoint; not used) |

### 5.2 Loading and normalisation

- TIFFs are read as float32; multi-channel images are averaged to one channel.
- Each image is divided by its own 99.5th-percentile intensity (the paper white) and clipped to [0, 1]. This removes exposure differences between the two cameras and between samples.
- Pairs are formed by identical file names in `CAM01` and `CAM02`; a regular expression restricts files to the selected level and extracts font and sample number.
- **Sanity check:** on 10 random pairs the mean gradient energy (after a σ = 2 px smoothing so sensor noise is not counted) must be higher for CAM01 than for CAM02; otherwise the notebook stops. For Level 1 the values were 0.000514 (GT) vs 0.000256 (input).

### 5.3 Train / validation / test split

Per font, the sorted sample IDs are shuffled with seed 42; the first 20 go to test, the next 5 to validation, the remaining 75 to training → **150 train / 10 validation / 40 test**, saved to `split.json` and re-used on every run. The notebook asserts the three sets are disjoint. Because the seed and ID lists are identical, **the same sample numbers form the test set at every level** (verified across Levels 0–9), which makes levels directly comparable.

### 5.4 Registration of ground truth to input

Pixel-wise losses and PSNR/SSIM need the two cameras aligned. The notebook does this once per level:

1. **Blur-size estimate.** For 4 training pairs, the sharp image is Gaussian-blurred with σ ∈ {{5, 10, …, 100}} px and a 1600 × 600 px central template is matched against the real blurred image (normalised cross-correlation, ±60 px search). The σ with the highest score is kept; the median over the 4 pairs is the level's `BLUR_SIGMA`. This number is a Gaussian-equivalent size used to drive later choices, not a physical measurement.
2. **Patch matching.** On 12 training pairs, the blurred-sharp image is cut into 320 × 160 px templates on an 80 px grid (blank regions with standard deviation < 0.08 are skipped). Each is located in the real input within ±60 px; matches with NCC ≥ 0.8 are kept (relaxed to 0.7, then 0.6, if fewer than 50 survive).
3. **One global affine.** `cv2.estimateAffine2D` with RANSAC (3 px threshold, 5000 iterations) fits a single 2 × 3 affine for the level, using training images only. For Level 1: 2970 matches, 2954 inliers.
4. **Warp and trim.** Every ground-truth image is warped into the input's frame (bicubic, replicated border) and 48 px are trimmed from each edge. The input image is never resampled.
5. **Check.** For all 200 pairs the correlation between blurred-GT and input over the text area is logged before and after warping (`pairing_checks.csv`). If registration fails entirely the previous level's affine is used as a fallback (never triggered).

{reg_table()}
![Registration](figures/registration_by_level.png)

The affine scale grows steadily with the level (0.993 → 1.057 in x): defocusing changes the magnification of CAM02. Alignment quality falls as blur grows, from 0.970 at Level 1 to 0.780 at Level 9. This matters twice: mis-aligned targets blur the training signal, and they lower the PSNR/SSIM that a perfect restoration could reach.

### 5.5 Model set-up and freezing

The model is built with the configuration in §4, the pretrained weights are loaded with `strict=True`, and `requires_grad` is switched off for every parameter whose name starts with one of the seven encoder-path prefixes (§4.4). Printed by the notebook: *Frozen 3.79M | Trainable 22.34M*.

### 5.6 Patch sampling and augmentation

- A random training image and a random position are drawn; the crop is accepted only if the ground-truth patch has standard deviation > 0.05 (i.e. it contains text). Up to 20 attempts are made, so nearly every patch contains strokes rather than blank paper.
- Augmentation: random rotation by 0/90/180/270° and a horizontal flip with probability 0.5, applied identically to input and target.
- Patch size and batch follow the estimated blur: **σ ≤ 30 px → 128 px × 4; 30 < σ ≤ 50 → 192 px × 2; σ > 50 → 256 px × 2** (`PATCH_OVERRIDE` can force a size). Levels 0–6 therefore used 128 × 4 and Levels 7–9 used 192 × 2.
- Images are held in RAM as float16 (about 2.5 GB per level) and patches are cut on the fly; there is no DataLoader and no fixed epoch. 3000 iterations × batch = 12,000 patches (Levels 0–6) or 6,000 patches (Levels 7–9).

### 5.7 Optimisation

| Setting | Value |
|:--|:--|
| Loss | Charbonnier: `mean( sqrt( (Î − GT)² + ε² ) )`, ε = 1e-3 — a smooth L1 loss that is less sensitive to outliers (mis-registered pixels, sensor noise) than L2 |
| Optimiser | AdamW, learning rate 2e-5, weight decay 1e-4, betas (0.9, 0.999), trainable parameters only |
| Schedule | Cosine annealing over 3000 iterations down to 1e-7 |
| Iterations | 3000 |
| Mixed precision | `torch.autocast` float16 forward pass + `GradScaler`; the loss is computed in float32 |
| Gradient clipping | Global norm 1.0 (after un-scaling) |
| Validation | Every 500 iterations and at iteration 0, on the 10 validation images, full-image tiled inference, PSNR and SSIM |
| Model selection | Checkpoint with the highest **validation SSIM** is saved as `levelN_best.pth`; the test set is never used for selection |
| Logging | Mean loss of each 50 iterations → `training_history.json`; validation lines → `train_log.txt` |

The learning rate is deliberately small (the original Restormer training uses 3e-4) because this is fine-tuning from good weights on few images.

{train_table()}
![Training curves](figures/training_curves_all_levels.png)

Most of the improvement arrives in the first 500 iterations at every level except 9; after that the curves flatten. The converged loss rises with the blur level (0.027 at Levels 0–1 to 0.085 at Level 9). At Level 9 the validation curve is not monotonic and the loss is still decreasing at the end.

### 5.8 Full-image inference

Test images (2264 × 1364) do not fit through the network in one pass on a T4, so `restore()` works in tiles:

- reflect-pad so height and width are multiples of 8;
- 768 × 768 px tiles with 192 px overlap (stride 576; 256 px overlap would be used for σ > 50, which no trained level reached), with the last tile in each direction snapped to the image edge;
- each tile is weighted by a separable window that ramps linearly from 0.05 to 1 over the outer 96 px, and overlapping tiles are blended by weighted average, so no seams are visible;
- the three output channels are averaged back to one channel and clipped to [0, 1];
- outputs are saved as 16-bit PNGs.

---

## 6. Evaluation method and metric definitions

Four versions of each test image are scored: **Blurred input** (CAM02), **Pretrained Restormer** (original weights, no fine-tuning — the same reference at every level), **Fine-tuned Restormer** (best-validation checkpoint of that level), and **Sharp GT** (the registered CAM01 image; OCR only).

| Metric | Definition | Source |
|:--|:--|:--|
| PSNR (dB) | `10·log10(1 / MSE)` between output and registered ground truth, data range 1.0, whole 2264 × 1364 image | Notebook (`skimage`) |
| SSIM | Structural similarity, `skimage` defaults (7 × 7 uniform window), data range 1.0, whole image | Notebook |
| Character accuracy | `max(0, 1 − Levenshtein(OCR, GT) / len(GT))` after collapsing whitespace | Notebook |
| Token accuracy | Fraction of ground-truth whitespace-separated tokens that appear in the OCR output (each OCR token can be used once; order ignored) | Notebook |
| CER | `Levenshtein(OCR, GT) / len(GT)` (not capped; equals 1 − character accuracy whenever that is ≥ 0) | Computed for this report from the saved transcripts |
| WER | Word-level Levenshtein distance / number of GT words (can exceed 100% when the OCR output has extra words) | Computed for this report |
| Mean edit distance | Average number of character edits per image (GT strings are 30–32 characters) | Computed for this report |
| Exact-match rate | Share of images whose OCR string equals the GT string exactly | Computed for this report |
| Empty-OCR rate | Share of images for which EasyOCR returned no text | Computed for this report |

**OCR procedure.** EasyOCR (`Reader(['en'], gpu=True)`, default detection and recognition models) is run on the 8-bit image with `detail=1, paragraph=False`. Detected boxes are grouped into lines by vertical position (boxes whose centres differ by less than half a box height share a line), sorted left-to-right within a line, and joined with spaces. No language model, dictionary or post-correction is used.

The character accuracies recomputed from the saved transcripts match the notebook's stored values to machine precision, and PSNR/SSIM recomputed from the saved images for four images per level match to within 0.003 dB and 0.00002 SSIM.

---

## 7. Results across levels

### 7.1 Full metric table

{summary_table()}
### 7.2 Gains of the fine-tuned model

{gains_table()}
![Gains](figures/gains_fine_tuned_vs_blurred.png)

### 7.3 By font

![Font breakdown](figures/font_breakdown.png)

{font_table(range(10))}
### 7.4 Spread across images

![Character accuracy distribution](figures/char_acc_distribution.png)

For Levels 2–5 nearly all images sit above 90%. From Level 6 the spread widens sharply (standard deviation 8.8 points at Level 6, 21.9 points at Level 9), which is mostly the Times/Verdana split shown above.

---

## 8. Level-by-level report

Each sub-section gives the configuration, validation trajectory, test metrics, font breakdown, the image comparison (blurred | fine-tuned | ground truth) with per-image scores and OCR output, and observations.

""")
for L in range(10): R.append(level_section(L)+"\n---\n")
R.append(f"""
## 9. Discussion

### 9.1 Why the pretrained model fails and fine-tuning works

The official defocus weights were trained on natural photographs with mild, spatially varying defocus. Here the blur is a large uniform disc on black-on-white glyphs. With no fine-tuning the model changes the image very little: PSNR is at best 0.05 dB above the input (and up to 1.1 dB below it) and OCR accuracy is essentially unchanged. After 3000 iterations on 150 images the same network gains 2.4–8.4 dB. The first 500 iterations deliver most of that, which indicates the pretrained decoder is close to a useful solution and mainly needs to learn the blur scale and the glyph prior of this domain.

### 9.2 Why restored images can beat the sharp ground truth on OCR

EasyOCR reads the ground-truth images at only 90.8–94.1% character accuracy and 55–69% token accuracy. Its typical error on these images is dropping or splitting wide glyphs such as W and V in the large, high-contrast, slightly noisy camera rendering. The fine-tuned outputs at Levels 2–5 are smoother and noise-free, and are read at 96.1–98.3%. Two consequences: the "Sharp GT (upper bound)" row should be read as "how well this recogniser does on a clean camera image", and absolute OCR percentages are specific to EasyOCR's default models. A recogniser trained on this font size would shift all rows upward.

### 9.3 Why accuracy falls from Level 6

Three things change together as blur grows: (a) information is physically lost — strokes thinner than the blur disc leave almost no trace, which hits serif Times harder than uniform-stroke Verdana; (b) registration becomes less certain (alignment score 0.87 at Level 6, 0.78 at Level 9), so the training target is itself slightly misplaced; (c) the training patch covers fewer blur widths (128 px at σ ≈ 30 is about 4σ; 192 px at σ ≈ 47 is also about 4σ), while the patch count halves from Level 7. The step between Level 5 and Level 6 (96.9% → 83.2%) coincides with (c) at its tightest under the 128 px rule.

### 9.4 PSNR/SSIM and OCR measure different things

- Level 1 has the highest fine-tuned PSNR (28.87 dB) but a *negative* OCR change.
- Level 5 has a moderate PSNR (21.98 dB) and the largest OCR change (+96.9 points).
- Level 8 input SSIM (0.417) is higher than Level 4 input SSIM (0.373) although Level 8 is far more blurred.

Image metrics are computed against a noisy, imperfectly registered camera image, so they saturate low (best SSIM ≈ 0.69) and reward matching background statistics. For a recognition-oriented project the OCR columns are the primary outcome; PSNR/SSIM are best used *within* a level to compare methods.

### 9.5 What the per-level design does and does not show

Each model is trained and tested on one blur level, so these results describe the **non-blind, level-matched** case: the blur level is known and a matching model is applied. They do not show how a model behaves on a level it was not trained on. The planned combined model (§11) is the experiment that addresses blind deblurring across levels.

---

## 10. Limitations and caveats

1. **Level 0 OCR is missing** from the export; its image metrics were recomputed for this report.
2. **Level 5 is a legacy run** with an incomplete manifest and no text log; its patch/batch setting is inferred from the notebook.
3. **σ estimates are coarse** (5 px grid) and Levels 0–2 sit on the grid minimum.
4. **One run per level, one seed.** No repeated runs, so there are no confidence intervals across trainings; the ± values in the tables are spread across the 40 test images.
5. **Small validation set** (10 images) for checkpoint selection; at Level 9 the selected checkpoint (iteration 2000) differs noticeably from its neighbours.
6. **Different patch/batch settings** for Levels 0–6 and 7–9 mean the levels are not trained under strictly identical budgets (12,000 vs 6,000 patches).
7. **Same fonts, same size, same display** in train and test. The text strings differ, but generalisation to other fonts, sizes or real documents is untested.
8. **Single recogniser.** All OCR numbers are EasyOCR with default models; token accuracy ignores word order and line breaks.
9. **Image metrics depend on registration**, which degrades with blur; part of the PSNR drop at high levels is measurement, not restoration.
10. **Ground truth is a camera image, not a rendered glyph**, so it contains sensor noise that a good restoration will (correctly) not reproduce; this caps SSIM.
11. **Earlier Phase-1 figures** (small "focusStep" subset, levels 0–4, different split and notebook) are not comparable with the numbers in this report and are superseded by it for those levels.

---

## 11. Level 10 and the combined "ultimate" model

**Level 10.** `HDC2021_step_10.zip` is in the folder and `level_10/manifest.json` reads *"pending - dataset not uploaded yet"*. No per-level training was run.

**Combined model.** `ultimate_model/` holds the start of a single model trained on Levels 5–10 together:

| Item | Value (from `ultimate_model/logs/train_log.txt` and `preprocessing_summary.csv`) |
|:--|:--|
| Start | 20 October 2026, Tesla T4 |
| Levels | 5, 6, 7, 8, 9, 10 — 200 pairs each, 150 / 10 / 40 split per level |
| Registration | Per-level global affine plus per-image refinement where it helped (87 images at Level 5, 20 at Level 6, none above) |
| Progressive patch stages | (until iteration 2500: 160 px, batch 4), (until 5000: 224 px, batch 2), (until 6000: 288 px, batch 2) |
| Learning rate | 8e-5, encoder at half that rate (encoder not frozen) |
| Pretrained baseline at iteration 0 | Validation 14.78 dB / SSIM 0.4396 (per level SSIM: L5 0.422, L6 0.417, L7 0.415, L8 0.457, L9 0.458, L10 0.469) |
| What exists | `checkpoints/ultimate_last.pth` (418.8 MB), registration cache, splits, one data-check figure |
| What does not exist | No best checkpoint, no validation lines after iteration 0, no metrics, no restored images |

| Level | Blur σ (px) | Matches | RANSAC inliers | NCC before | NCC after |
|--:|--:|--:|--:|--:|--:|
| 5 | 25 | 2950 | 1581 | 0.792 | 0.892 |
| 6 | 30 | 2834 | 915 | 0.758 | 0.867 |
| 7 | 37 | 2584 | 485 | 0.722 | 0.836 |
| 8 | 45 | 2405 | 249 | 0.669 | 0.804 |
| 9 | 47 | 2060 | 148 | 0.626 | 0.780 |
| 10 | 55 | 1745 | 82 | 0.559 | 0.689 |

The log stops immediately after the first training stage begins, so this model has to be treated as **not yet trained/evaluated**. The inlier counts confirm the registration trend from §5.4: at Level 10 only 82 of 1745 matches agree with a single affine.

---

## 12. How to reproduce or extend

1. Put `HDC2021_stepN.zip` in `FinalYearProject-Phase2/level_NN/dataset/` on Drive and keep `pretrained/single_image_defocus_deblurring.pth` in place.
2. Open the notebook in Colab with a T4 GPU, set `LEVEL = N` in the first cell, leave `INIT_FROM = 'pretrained'`, and *Run all*. A level takes about 45 minutes of training at 128 × 4 and about 85 minutes at 192 × 2, plus registration, baseline inference and OCR.
3. Outputs appear under `level_NN/`: checkpoints, `results/` (split, registration, CSVs, OCR transcripts, restored PNGs, figures), `logs/`, and an updated `manifest.json`.
4. The last cell rebuilds `all_levels_summary.csv` and `all_levels_trend.png` from every level that has a summary.

Suggested next runs, in order of expected value:

- **Level 0 OCR:** re-run the evaluation cells for Level 0 (the checkpoint and split exist; `RESUME=True` skips training) to fill the missing row.
- **Level 6 with `PATCH_OVERRIDE = (192, 2)`** to test the patch-size explanation in §9.3.
- **Level 9 with more iterations** (the loss has not converged) and/or `INIT_FROM = 'previous'` (start from Level 8's best checkpoint).
- **Level 10** per-level run for completeness.
- **Cross-level test:** apply each level's checkpoint to the neighbouring levels' test sets to quantify how blind the per-level models are.
- **Finish and evaluate the combined model.**
- **A second recogniser** (for example TrOCR, already used elsewhere in the project) to check that the OCR conclusions do not depend on EasyOCR.

---

## 13. Appendix

### 13.1 Files produced with this report

| Path | Content |
|:--|:--|
| `REPORT.md` / `REPORT.docx` / `REPORT.pdf` / `REPORT.html` | This report |
| `PROJECT_CONTEXT.md` | Self-contained context file for a reader or LLM with no prior knowledge |
| `tables/COMPARISON_TABLES.md` | All comparison tables in one Markdown file |
| `tables/all_levels_summary_extended.csv` | Mean, standard deviation, min and max of every metric per level and method |
| `tables/all_levels_per_image_metrics.csv` | Every test image × method: PSNR, SSIM, character/token accuracy, CER, WER, GT text, OCR text |
| `tables/all_levels_by_font.csv` | Per-font means |
| `tables/gains_fine_tuned.csv` | Gains and per-image win/tie/loss counts |
| `tables/training_and_registration_by_level.csv` | Blur σ, patch/batch, validation start/best, loss, wall time, registration quality, affine |
| `tables/all_levels_ocr_outputs.csv` | All 1,440 OCR transcripts with error metrics |
| `comparisons/level_NN_comparison.jpg` | 4 × 3 grid: blurred, fine-tuned, ground truth, with metrics and OCR text |
| `comparisons/level_NN_closeup.jpg` | Centre crops of the same images at native resolution |
| `images/level_NN/` | Individual full-resolution panels (blurred, pretrained, fine-tuned, ground truth) for the four displayed images |
| `figures/` | Architecture diagram (PNG and SVG), block detail, pipeline, charts |
| `source_results/` | Copies of the original per-level CSVs, logs, manifests and notebook figures from the Drive export |
| `scripts/` | The Python scripts that produced the tables and figures |

### 13.2 Glossary

- **CAM01 / CAM02** — the in-focus and the defocused camera; ground truth and input.
- **Focus step / level** — one defocus setting of CAM02 in HDC 2021; higher is blurrier.
- **σ (blur sigma)** — width of the Gaussian that best mimics the level's blur for matching purposes.
- **NCC** — normalised cross-correlation, the alignment score (1 = identical up to brightness/contrast).
- **MDTA / GDFN** — the attention and feed-forward sub-layers of a Restormer block (§4.2).
- **Charbonnier loss** — smooth approximation of the absolute error.
- **AMP** — automatic mixed precision (float16 forward pass).
- **CER / WER** — character / word error rate.

### 13.3 References

- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang. *Restormer: Efficient Transformer for High-Resolution Image Restoration.* CVPR 2022. Code and weights: github.com/swz30/Restormer.
- M. Juvonen, S. Siltanen, F. Silva de Moura. *Helsinki Deblur Challenge 2021: description of photographic data.* 2021 (dataset on Zenodo).
- JaidedAI. *EasyOCR.* github.com/JaidedAI/EasyOCR.
""")
open(f'{OUT}/REPORT.md','w').write('\n'.join(R))
# comparison tables file
C=f"""# Comparison tables — Restormer fine-tuning, HDC 2021 blur Levels 0–9

All values are test-set means over 40 images per level unless stated. Level 0 has no OCR results in the export; its PSNR/SSIM were recomputed from the saved restored images. See `REPORT.md` for definitions and discussion; the same data are in the CSV files in this folder.

## 1. Headline: blurred input vs fine-tuned Restormer

{headline_table()}
## 2. All methods, all metrics

{summary_table()}
## 3. Gains of the fine-tuned model

{gains_table()}
## 4. By font

{font_table(range(10))}
## 5. Training by level

{train_table()}
## 6. Registration by level

{reg_table()}
## 7. Image comparison grids

One grid per level: blurred input (left), fine-tuned Restormer (middle), ground truth (right), four test images each, with PSNR, SSIM and OCR output.

"""
for L in range(10): C+=f"### Level {L}\n\n![Level {L}](../comparisons/level_{L:02d}_comparison.jpg)\n\n![Level {L} close-up](../comparisons/level_{L:02d}_closeup.jpg)\n\n"
open(f'{OUT}/tables/COMPARISON_TABLES.md','w').write(C)
print(len('\n'.join(R)), len(C))
