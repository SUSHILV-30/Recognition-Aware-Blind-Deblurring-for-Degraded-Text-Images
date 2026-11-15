# OCR Results — Restormer, fine-tuned

How readable the restored pages are, measured with OCR rather than PSNR/SSIM. Phase 2
scores every level from 1 to 9 with the same reader, the same 40-page test split and the
same scoring, so the levels can be compared directly. Generated from
[`report/tables/all_levels_summary_extended.csv`](report/tables/all_levels_summary_extended.csv)
and [`all_levels_by_font.csv`](report/tables/all_levels_by_font.csv).

## Protocol (Phase 2)

| | |
|---|---|
| Reader | EasyOCR (`Reader(['en'])`, default models, `detail=1, paragraph=False`); boxes grouped into lines by y, sorted by x, joined by spaces |
| Test set | 40 pages per level (20 Times New Roman, 20 Verdana), the same sample IDs at every level, seed 42 |
| Ground truth for OCR | The `.txt` file shipped with each HDC 2021 page |
| Character accuracy | `max(0, 1 − Levenshtein(pred, GT) / len(GT))` |
| Token accuracy | Share of ground-truth tokens found among the OCR tokens, order ignored |
| CER, WER, exact match, empty OCR | Added afterwards from the saved transcripts (`results/phase2/levelN/ocr_outputs.csv`) |
| Code | [`notebooks/05_phase2_per_level_0_to_9.ipynb`](notebooks/05_phase2_per_level_0_to_9.ipynb); Level 5 from [`02_phase2_level5_restormer_finetune.ipynb`](notebooks/02_phase2_level5_restormer_finetune.ipynb) |

Level 0 has no OCR: the evaluation cells were not saved for that run.

## Results, Levels 1–9 (%)

### Character accuracy

| Level | Blurred input | Pretrained | **Fine-tuned** | Sharp GT |
|--:|--:|--:|--:|--:|
| 1 | 98.4 | 98.1 | **92.6** | 90.8 |
| 2 | 94.3 | 95.0 | **96.1** | 92.2 |
| 3 | 75.5 | 75.5 | **98.3** | 92.5 |
| 4 | 23.5 | 27.3 | **97.4** | 91.3 |
| 5 | 0.0 | 0.0 | **96.9** | 94.1 |
| 6 | 0.0 | 0.0 | **83.2** | 93.4 |
| 7 | 0.0 | 0.0 | **77.0** | 91.6 |
| 8 | 0.0 | 0.0 | **73.5** | 92.6 |
| 9 | 0.0 | 0.0 | **52.8** | 93.2 |

### Token accuracy

| Level | Blurred input | Pretrained | **Fine-tuned** | Sharp GT |
|--:|--:|--:|--:|--:|
| 1 | 85.8 | 83.7 | **60.5** | 57.4 |
| 2 | 63.5 | 65.0 | **75.0** | 55.6 |
| 3 | 24.0 | 23.5 | **86.5** | 63.8 |
| 4 | 0.5 | 0.5 | **82.2** | 55.1 |
| 5 | 0.0 | 0.0 | **75.0** | 64.7 |
| 6 | 0.0 | 0.0 | **32.0** | 69.1 |
| 7 | 0.0 | 0.0 | **22.3** | 61.7 |
| 8 | 0.0 | 0.0 | **23.9** | 63.8 |
| 9 | 0.0 | 0.0 | **10.9** | 66.8 |

### Character error rate (lower is better)

| Level | Blurred input | Pretrained | **Fine-tuned** | Sharp GT |
|--:|--:|--:|--:|--:|
| 1 | 1.6 | 1.9 | **7.4** | 9.2 |
| 2 | 5.8 | 5.0 | **3.9** | 7.8 |
| 3 | 24.5 | 24.5 | **1.7** | 7.5 |
| 4 | 76.5 | 72.7 | **2.6** | 8.7 |
| 5 | 100.0 | 100.0 | **3.1** | 5.9 |
| 6 | 100.0 | 100.0 | **16.8** | 6.6 |
| 7 | 100.0 | 100.0 | **23.0** | 8.4 |
| 8 | 100.0 | 100.0 | **26.5** | 7.4 |
| 9 | 100.0 | 100.0 | **47.2** | 6.8 |

### Word error rate (lower is better)

| Level | Blurred input | Pretrained | **Fine-tuned** | Sharp GT |
|--:|--:|--:|--:|--:|
| 1 | 15.9 | 19.6 | **56.9** | 65.0 |
| 2 | 42.7 | 39.8 | **27.5** | 58.5 |
| 3 | 88.0 | 93.3 | **14.2** | 49.9 |
| 4 | 130.2 | 135.0 | **17.8** | 65.9 |
| 5 | 100.0 | 100.0 | **25.8** | 51.5 |
| 6 | 100.0 | 100.0 | **78.8** | 38.7 |
| 7 | 100.0 | 100.0 | **98.1** | 61.3 |
| 8 | 100.0 | 100.0 | **98.4** | 56.5 |
| 9 | 100.0 | 100.0 | **133.1** | 50.2 |

### Exact match and empty OCR

| Level | Exact match, blurred | Exact match, fine-tuned | Empty OCR, blurred | Empty OCR, fine-tuned |
|--:|--:|--:|--:|--:|
| 1 | 55.0 | 20.0 | 0.0 | 0.0 |
| 2 | 22.5 | 37.5 | 0.0 | 0.0 |
| 3 | 2.5 | 62.5 | 0.0 | 0.0 |
| 4 | 0.0 | 47.5 | 0.0 | 0.0 |
| 5 | 0.0 | 40.0 | 100.0 | 0.0 |
| 6 | 0.0 | 0.0 | 100.0 | 0.0 |
| 7 | 0.0 | 0.0 | 100.0 | 0.0 |
| 8 | 0.0 | 0.0 | 100.0 | 0.0 |
| 9 | 0.0 | 0.0 | 100.0 | 0.0 |

### Character accuracy by font

| Level | Fine-tuned, Times | Fine-tuned, Verdana | Sharp GT, Times | Sharp GT, Verdana |
|--:|--:|--:|--:|--:|
| 1 | 89.5 | 95.8 | 87.3 | 94.3 |
| 2 | 93.9 | 98.3 | 88.7 | 95.6 |
| 3 | 99.1 | 97.5 | 89.8 | 95.2 |
| 4 | 96.9 | 97.9 | 88.5 | 94.0 |
| 5 | 96.7 | 97.2 | 92.1 | 96.1 |
| 6 | 78.5 | 87.9 | 91.7 | 95.1 |
| 7 | 66.3 | 87.7 | 87.9 | 95.3 |
| 8 | 57.5 | 89.5 | 91.4 | 93.9 |
| 9 | 34.3 | 71.3 | 90.3 | 96.2 |

## What the numbers say

- **There is a usefulness window.** At Levels 1–2 the blurred page is already readable
  (98.4% and 94.2%) and restoration changes little; at Level 1 it costs 5.7 points. From
  Level 2 to Level 5 the fine-tuned output reads at 96–98%. Levels 6–8 reach 73–83% of
  characters but only 22–32% of tokens, and Level 9 falls to 52.8%.
- **Restoration reads pages OCR could not read at all.** From Level 5 EasyOCR finds no text
  on the blurred page or on the pretrained output (100% empty). After fine-tuning it reads
  96.9% of characters at Level 5.
- **The pretrained checkpoint does not help OCR.** It stays within 4 points of the blurred
  input at every level.
- **"Sharp GT" is not a ceiling.** EasyOCR reads the sharp camera images at 90.8–94.1%, and
  the fine-tuned outputs at Levels 2–5 score higher (96.1–98.3%): the restored strokes are
  cleaner and higher-contrast than the grainy capture.
- **Times New Roman fails first.** From Level 6 the serif font degrades much faster: at
  Level 9 the fine-tuned output reads 34.3% of Times characters against 71.3% for Verdana.
- **Token accuracy falls faster than character accuracy.** Each token is a random
  10-character string, so one wrong character loses the token.

## Phase 1 baselines (focusStep, superseded)

Before Phase 2, OCR was scored on the focusStep pages with [`scripts/ocr_eval.py`](scripts/ocr_eval.py):
40 pages per level, with sharp captures for only 16 pages at Levels 1–2. These baselines use a
different page set from Phase 2, so they are kept for the record only.

### Character accuracy (%)

| Level | Blurred input | Restormer, pretrained | Restormer, fine-tuned | NAFNet | MPRNet | Restormer, scratch | HDC U-Net | Sharp GT (upper bound) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 94.4 | — | — | — | — | — | — | 88.6 |
| 2 | 91.2 | — | — | — | — | — | — | 86.8 |
| 3 | 66.0 | — | — | — | — | — | — | 88.5 |
| 4 | 20.0 | — | — | — | — | — | — | 89.5 |
| 5 | 0.0 | 0.0 | **96.9** | — | — | — | — | 94.1 |
| 6 | 0.0 | 0.0 | **83.2** | — | — | — | — | 93.4 |

### Token accuracy (%)

| Level | Blurred input | Restormer, pretrained | Restormer, fine-tuned | NAFNet | MPRNet | Restormer, scratch | HDC U-Net | Sharp GT (upper bound) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 61.3 | — | — | — | — | — | — | 43.8 |
| 2 | 51.5 | — | — | — | — | — | — | 44.8 |
| 3 | 11.9 | — | — | — | — | — | — | 40.3 |
| 4 | 0.0 | — | — | — | — | — | — | 44.3 |
| 5 | 0.0 | 0.0 | **75.0** | — | — | — | — | 64.7 |
| 6 | 0.0 | 0.0 | **32.0** | — | — | — | — | 69.1 |

`—` means not measured yet.

### By font, character accuracy (%)

| Level | Method | Times New Roman | Verdana |
|---|---|---:|---:|
| 1 | Blurred input | 94.2 | 94.5 |
| 1 | Sharp GT | 86.7 (n = 9) | 91.1 (n = 7) |
| 2 | Blurred input | 86.8 | 95.6 |
| 2 | Sharp GT | 82.8 (n = 9) | 91.9 (n = 7) |
| 3 | Blurred input | 49.9 | 82.1 |
| 3 | Sharp GT | 87.7 | 89.2 |
| 4 | Blurred input | 10.1 | 30.0 |
| 4 | Sharp GT | 85.9 | 93.1 |
| 5 | Restormer, fine-tuned | 96.7 | 97.2 |
| 5 | Sharp GT | 92.1 | 96.1 |
| 6 | Restormer, fine-tuned | 78.5 | 87.9 |
| 6 | Sharp GT | 91.7 | 95.1 |

### What the Phase 1 baselines say

- **OCR fails at Level 3, not Level 1.** EasyOCR still reads 91–94% of characters on the
  blurred page at Levels 1–2. It drops to 66% at Level 3, 20% at Level 4 and 0% from Level 5.
  Restoration matters most from Level 3 up.
- **Fine-tuned Restormer reads pages that OCR could not read at all.** At Level 5 it takes
  character accuracy from 0% to 96.9%. At Level 6 it reaches 83.2%. The pretrained checkpoint
  stays at 0% on both levels, even though it raises SSIM.
- **Token accuracy falls faster than character accuracy.** Each token is a 10-character random
  string, so one wrong character loses the whole token. At Level 6 that is 83.2% of characters
  but only 32.0% of tokens.
- **Times New Roman suffers first.** Thin serif strokes vanish sooner than Verdana's. At Level 3
  the blurred Times page reads 49.9% against 82.1% for Verdana, and the same gap shows after
  restoration at Level 6 (78.5% vs 87.9%).
- **"Sharp GT" is not 100%, and sometimes not the top score.** The text is random characters
  (l/I, 0/O, 5/S), so there is no dictionary to correct against. At Levels 1–2 the mildly
  blurred page scores slightly above the sharp capture on the same 16 pages (94.1% vs 88.6%),
  because light blur smooths sensor noise and EasyOCR drops some thin strokes on the raw sharp
  image. At Level 5 the restored output also beats the sharp capture (96.9% vs 94.1%) for the
  same reason: it is cleaner and higher-contrast.


Earlier OCR numbers in [`docs/results.md`](docs/results.md) (51.0% token, 83.3% character)
came from a Swin2SR → EasyOCR → TrOCR pipeline with no Restormer stage. They use a different
recognizer, so they are not included in the tables above.

## Files

- [`results/phase2/all_levels_summary.csv`](results/phase2/all_levels_summary.csv): one row per level and method, Levels 1–9
- [`results/phase2/levelN/`](results/phase2/): per level `levelN_summary.csv`, `levelN_per_image.csv`, `levelN_by_font.csv` and `ocr_outputs.csv` with every transcription
- [`report/tables/`](report/tables/): the same numbers with CER, WER, exact match, spread and gains
- [`results/ocr/levels_1_4/`](results/ocr/levels_1_4/): Phase 1 focusStep baselines
