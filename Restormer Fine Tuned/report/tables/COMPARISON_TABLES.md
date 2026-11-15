# Comparison tables — Restormer fine-tuning, HDC 2021 blur Levels 0–9

All values are test-set means over 40 images per level unless stated. Level 0 has no OCR results in the export; its PSNR/SSIM were recomputed from the saved restored images. See `REPORT.md` for definitions and discussion; the same data are in the CSV files in this folder.

## 1. Headline: blurred input vs fine-tuned Restormer

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

## 2. All methods, all metrics

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

## 3. Gains of the fine-tuned model

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

## 4. By font

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

## 5. Training by level

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

## 6. Registration by level

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

## 7. Image comparison grids

One grid per level: blurred input (left), fine-tuned Restormer (middle), ground truth (right), four test images each, with PSNR, SSIM and OCR output.

### Level 0

![Level 0](../comparisons/level_00_comparison.jpg)

![Level 0 close-up](../comparisons/level_00_closeup.jpg)

### Level 1

![Level 1](../comparisons/level_01_comparison.jpg)

![Level 1 close-up](../comparisons/level_01_closeup.jpg)

### Level 2

![Level 2](../comparisons/level_02_comparison.jpg)

![Level 2 close-up](../comparisons/level_02_closeup.jpg)

### Level 3

![Level 3](../comparisons/level_03_comparison.jpg)

![Level 3 close-up](../comparisons/level_03_closeup.jpg)

### Level 4

![Level 4](../comparisons/level_04_comparison.jpg)

![Level 4 close-up](../comparisons/level_04_closeup.jpg)

### Level 5

![Level 5](../comparisons/level_05_comparison.jpg)

![Level 5 close-up](../comparisons/level_05_closeup.jpg)

### Level 6

![Level 6](../comparisons/level_06_comparison.jpg)

![Level 6 close-up](../comparisons/level_06_closeup.jpg)

### Level 7

![Level 7](../comparisons/level_07_comparison.jpg)

![Level 7 close-up](../comparisons/level_07_closeup.jpg)

### Level 8

![Level 8](../comparisons/level_08_comparison.jpg)

![Level 8 close-up](../comparisons/level_08_closeup.jpg)

### Level 9

![Level 9](../comparisons/level_09_comparison.jpg)

![Level 9 close-up](../comparisons/level_09_closeup.jpg)

