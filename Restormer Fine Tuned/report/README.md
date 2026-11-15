# Restormer_Report_Levels0-9

Analysis of the FYP Phase 2 Drive export (`FinalYearProject-Phase2`), the HDC 2021 dataset archives and the training notebook. Generated 12 November 2026.

| Open this | For |
|:--|:--|
| `REPORT.pdf` (also `.html`, `.md`) | The full report: summary, dataset, Restormer architecture, training method, results for every level with image comparisons, discussion, limitations |
| `PROJECT_CONTEXT.md` | One self-contained file to hand to a new reader or an LLM with no prior context |
| `tables/COMPARISON_TABLES.md` | All comparison tables plus the image grids in one place |
| `tables/*.csv` | Every number: per-level summary (mean/std/min/max), per-image metrics with OCR text, by-font, gains, training and registration |
| `comparisons/level_NN_comparison.jpg` | Blurred (left) · fine-tuned (middle) · ground truth (right), 4 test images, with PSNR, SSIM, OCR accuracy and OCR text |
| `comparisons/level_NN_closeup.jpg` | Native-resolution centre crops of the same images |
| `figures/` | `restormer_architecture.png/.svg`, `transformer_block_detail.png`, `pipeline_overview.png`, trend/gain/training/registration/font/PSF charts |
| `../results/phase2/` | The original CSVs, logs, manifests and notebook figures from the export, one folder per level |
| `scripts/` | Python used to build the tables and figures. Each script sets its input and output paths at the top (`H`, `P2`, `OUT`, `TMP`); point them at the extracted Drive export and this folder before running |

Notes: Level 0 has no OCR results in the export (PSNR/SSIM recomputed from saved images). Level 10 and the combined "ultimate" model are not trained/evaluated. `REPORT.html` and the `.md` files load images by relative path, so keep them inside this folder.

In this repository the individual full-resolution panels (`images/`, 97 MB) and `REPORT.docx` are left out; the comparison grids in `comparisons/` show the same images, and `REPORT.pdf` has the same content as the `.docx`.
