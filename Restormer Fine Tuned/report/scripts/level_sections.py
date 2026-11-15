from build_common import *
NOTES={
0:"""- **What this level is.** Step 0 is the mildest setting in the dataset: the "blurred" camera is only slightly out of focus, so the input already scores 26.08 dB / 0.524 SSIM against the ground truth.
- **Restoration quality.** Fine-tuning still helps: 28.74 dB / 0.691 SSIM (+2.66 dB, +0.167 SSIM), and every one of the 40 test images improves on both metrics. The un-tuned pretrained model *lowers* PSNR (24.98 dB), which is the same domain-mismatch pattern seen at every level.
- **Font gap.** Times outputs reach 30.41 dB while Verdana reaches 27.06 dB. The same gap is present in the blurred inputs (27.11 vs 25.05 dB), so it comes from the data (the Verdana pairs at this step differ more from their ground truth), not from the model.
- **Gap in the exported results.** The Level 0 folder has a best/last checkpoint, training log, split, registration files and all 80 restored PNGs, but **no `level0_summary.csv`, `level0_per_image.csv`, `ocr_outputs.csv` or `manifest.json`**. The OCR/summary cells were therefore not run (or not saved) for this level. The PSNR/SSIM values here were recomputed for this report from the saved restored images and the dataset, using the notebook's exact preprocessing; the same recomputation reproduces the notebook's own CSV values for Levels 1–9 to within 0.003 dB. **OCR accuracy for Level 0 is not available.**""",
1:"""- **Image quality improves strongly.** +5.70 dB PSNR and +0.207 SSIM over the blurred input; all 40 test images improve.
- **OCR accuracy goes down, not up.** EasyOCR already reads the Level 1 blurred input at 98.35% character accuracy. After restoration it reads 92.62% (−5.7 points); 28 of 40 images get worse, 11 tie and 1 improves.
- **Why.** The sharp ground truth itself only scores 90.81%. On the sharp, high-contrast renderings EasyOCR repeatedly drops wide glyphs (for example `pApReWCcGs` is read as `pApRe CcGs` on both the ground truth and the fine-tuned output, but correctly on the blurred input). The fine-tuned model makes the image look like the ground truth, so it inherits the recogniser's behaviour on ground-truth-like images. At this level the blur is too mild to hurt recognition, so deblurring has nothing to gain on the OCR metric.
- **Practical reading.** For Level 1 (and by extension Level 0) the restoration is a visual-quality improvement only; a recognition-driven pipeline should pass these images through unchanged.""",
2:"""- **First level where restoration helps OCR.** Character accuracy rises from 94.25% to 96.07% and token accuracy from 63.5% to 75.0%; 18 images improve, 17 tie, 5 get worse.
- **PSNR gain jumps to +8.01 dB** (19.37 → 27.38 dB), SSIM +0.225.
- **Blur estimate caveat.** The estimated σ is 5 px for Levels 0, 1 and 2 alike. 5 px is the smallest value in the search grid, so for these three levels the estimate is a floor, not a measurement; the PSF images in §3 show the blur does grow from step 0 to step 2.""",
3:"""- **Best OCR result of the whole study.** Fine-tuned character accuracy is 98.28% and token accuracy 86.5%, with 62.5% of the test images transcribed with zero errors. The blurred input reads 75.55% / 24.0%.
- **Largest PSNR gain.** +8.38 dB (17.34 → 25.72 dB). 38 of 40 images improve on character accuracy, 2 tie, none get worse.
- **Above the "upper bound".** The fine-tuned output is read more accurately than the sharp ground truth (98.28% vs 92.48%). The slightly smoother, noise-free strokes produced by the network suit EasyOCR better than the grainy camera image (see §9.2).""",
4:"""- **Blurred input is now unreadable in practice.** 23.5% character accuracy and 0.5% token accuracy on the input; the pretrained model does not change that (27.3% / 0.5%).
- **Restoration recovers it almost completely.** 97.40% character accuracy and 82.2% token accuracy; all 40 images improve. PSNR +8.03 dB, SSIM +0.222.
- **Registration starts to matter.** The alignment score after warping drops to 0.915 (from 0.948 at Level 3). The blur is now wide enough (σ ≈ 20 px) that matching the sharp image to the blurred one is less precise.""",
5:"""- **The headline level.** EasyOCR detects **no text at all** on any of the 40 blurred inputs or any of the 40 pretrained-Restormer outputs (0% accuracy, 100% empty outputs). After fine-tuning it reads 96.94% of characters and 75.0% of tokens, and 40% of images are transcribed perfectly.
- **Image metrics.** 15.35 → 21.98 dB (+6.63 dB), SSIM 0.372 → 0.582 (+0.210).
- **Legacy run.** Level 5 was trained first, with the earlier version of the notebook, before the Drive folder was reorganised. Its results sit one folder deeper (`level_05/results/level5_results/`), its `manifest.json` has no training configuration, and there is no `train_log.txt`. Blur σ ≈ 25 px comes from the notebook comment and from the combined-model preprocessing log; patch 128 × batch 4 follows from the notebook's rule for σ ≤ 30 and its comment that Level 5 used 128 × 4.""",
6:"""- **Clear drop from Level 5.** Character accuracy 83.22% (from 96.94% at Level 5) and token accuracy 32.0% (from 75.0%). No image is transcribed perfectly any more. The input is still completely unreadable (0%).
- **Image metrics.** 15.03 → 20.19 dB (+5.16 dB), SSIM +0.197.
- **A probable contributor: patch size.** σ ≈ 30 px sits exactly on the boundary of the notebook's rule, so Level 6 was trained with 128 px patches. A 128 px patch is only about 4σ wide, i.e. smaller than the blur's footprint (roughly ±2–3σ), so much of what the network needs to undo the blur lies outside the patch. Levels 7–9 use 192 px patches. Re-running Level 6 with `PATCH_OVERRIDE = (192, 2)` is a cheap experiment to check this.
- **Fonts diverge.** Times 78.5% vs Verdana 87.9% character accuracy; thin serif strokes are lost first.""",
7:"""- **Still usable at character level, weak at word level.** 77.04% character accuracy, 22.3% token accuracy; PSNR +5.16 dB (14.71 → 19.87 dB), SSIM +0.194.
- **Training setup changes here.** σ ≈ 37 px triggers 192 px patches with batch 2, so the model sees 6,000 patches instead of 12,000 and each iteration takes about twice as long (81 min total vs ~45 min).
- **Strong font effect.** Times 66.3% vs Verdana 87.7% character accuracy.""",
8:"""- **Similar to Level 7 on images, slightly lower on OCR.** 19.47 dB / 0.563 SSIM; 73.47% character accuracy and 23.9% token accuracy.
- **Font effect widens further.** Times 57.5% vs Verdana 89.5%. Verdana at Level 8 is read about as well as at Level 6–7; almost all of the loss is on the serif font.
- **Input SSIM is higher than at Level 7 (0.417 vs 0.373) although the blur is stronger.** The same jump appears for the pretrained model. SSIM against a noisy ground truth is dominated by background statistics rather than text here, so SSIM values should be compared within a level, not across levels (see §10).""",
9:"""- **The method is near its limit.** PSNR gain falls to +2.42 dB (14.39 → 16.81 dB), SSIM +0.112, character accuracy 52.82% and token accuracy 10.9%. Every image still improves over the (completely unreadable) input.
- **Training was unstable.** Validation PSNR moved 15.48 → 15.44 → 15.12 → 16.70 → 15.88 → 16.12 dB across checkpoints and the best checkpoint (by SSIM) is iteration 2000, not the last one. The loss is still falling at iteration 3000 (0.0845), so the model is under-trained at this level.
- **Alignment is the weakest of all levels.** Mean alignment score 0.780, worst pair 0.730. With σ ≈ 47 px the registration itself is uncertain, which caps both the training signal and the measured PSNR.
- **Outputs show invented strokes.** In the close-up figure the Times outputs contain blobs and broken strokes that are not in the ground truth; Verdana remains partly legible (71.3% vs 34.3% for Times).""",
}
def level_section(L):
    d=info[str(L)]; t=tr.loc[L]; S=[]
    S.append(f"### 8.{L+1} Level {L}\n")
    ck=f"`level_{L:02d}/checkpoints/level{L}_best.pth` (104.7 MB, weights only), `level{L}_last.pth` (283.7 MB, with optimiser state)"
    rows=[['Dataset archive',f"`HDC2021_step{'_' if L==10 else ''}{L}.zip` — 200 pairs (100 Times, 100 Verdana)"],['Split (same image IDs at every level)','150 train / 10 validation / 40 test'],
          ['Estimated blur σ',f"{int(t.blur_sigma_px)} px"+(' (grid minimum — treat as "≤ 5 px")' if t.blur_sigma_px==5 else '')+(' (from notebook comment and combined-model log)' if L==5 else '')],
          ['Registration alignment score (NCC)',f"{t.ncc_before:.3f} before → {t.ncc_after:.3f} after (worst pair {t.ncc_after_min:.3f})"],
          ['Global affine (input → GT)',f"scale x {t.affine_scale_x:.4f}, scale y {t.affine_scale_y:.4f}, shift ({t.affine_tx:.1f}, {t.affine_ty:.1f}) px"],
          ['Initial weights','Original pretrained Restormer (`single_image_defocus_deblurring.pth`); no curriculum from other levels'],
          ['Training patch × batch',f"{int(t.patch)} px × {int(t.batch)}"+(' (inferred, see notes)' if L==5 else '')],['Iterations / learning rate','3000 / 2e-5 (cosine decay to 1e-7)'],['Inference tiling','768 px tiles, 192 px overlap'],
          ['Best checkpoint',f"iteration {int(t.best_iter)} — validation {t.best_val_psnr:.2f} dB / SSIM {t.best_val_ssim:.4f}"],
          ['Run date / wall time',(f"{t.date} / {t.train_minutes:.0f} min on a Tesla T4" if isinstance(t.date,str) else 'October 2026 (legacy notebook; no log file)')],['Checkpoints',ck]]
    S.append(md(rows,['Item','Value'],['l','l']))
    h=d.get('hist')
    if h:
        S.append("\n**Validation during training** (10 held-out images, full-image tiled inference)\n\n"+md([['PSNR (dB)']+[f(x) for x in h['val_psnr']],['SSIM']+[f(x,4) for x in h['val_ssim']]],['Iteration']+[str(i) for i in h['val_iter']]))
    rows=[]
    for m in ORDER:
        a=A(L,m)
        if a is None: continue
        rows.append([m,f"{f(a.psnr)} ± {f(a.psnr_std)}" if not np.isnan(a.psnr) else '—',f"{f(a.ssim,4)} ± {f(a.ssim_std,4)}" if not np.isnan(a.ssim) else '—',f"{f(a.char_acc,2,True)} ± {f(a.char_acc_std,2,True)}" if not np.isnan(a.char_acc) else '—',f(a.char_acc_min,1,True)+' – '+f(a.char_acc_max,1,True) if not np.isnan(a.char_acc) else '—',f(a.token_acc,2,True),f(a.cer,2,True),f(a.wer,2,True),f(a.edit_dist,2),f(a.exact_match,1,True),f(a.empty_ocr,0,True)])
    S.append(f"\n**Test-set results** (n = 40; mean ± standard deviation across images)\n\n"+md(rows,['Method','PSNR (dB)','SSIM','Char acc %','Char acc range %','Token acc %','CER %','WER %','Mean edit distance','Exact-match %','Empty OCR %'],['l']+['r']*10))
    g=gains.loc[L]
    if L>0: S.append(f"\nFine-tuned vs blurred input: **{g.dpsnr_vs_blur:+.2f} dB PSNR, {g.dssim_vs_blur:+.4f} SSIM, {g.dchar_vs_blur_pts:+.1f} points character accuracy, {g.dtoken_vs_blur_pts:+.1f} points token accuracy.** Fine-tuned vs pretrained: {g.dpsnr_vs_pre:+.2f} dB, {g.dssim_vs_pre:+.4f} SSIM, {g.dchar_vs_pre_pts:+.1f} points character accuracy. Per image: PSNR improved on {int(g.psnr_wins)}/40, SSIM on {int(g.ssim_wins)}/40; character accuracy better / equal / worse than blurred on {int(g.char_wins)} / {int(g.char_ties)} / {int(g.char_losses)} images.\n")
    else: S.append(f"\nFine-tuned vs blurred input: **{g.dpsnr_vs_blur:+.2f} dB PSNR, {g.dssim_vs_blur:+.4f} SSIM.** Fine-tuned vs pretrained: {g.dpsnr_vs_pre:+.2f} dB, {g.dssim_vs_pre:+.4f} SSIM. PSNR improved on {int(g.psnr_wins)}/40 images, SSIM on {int(g.ssim_wins)}/40.\n")
    S.append("**By font**\n\n"+font_table([L]).replace('| Level ','| L ',1))
    S.append(f"\n**Image comparison — blurred (left), fine-tuned (middle), ground truth (right)**\n\nThe four images are test-split positions 1, 11, 21 and 31 (two Times, two Verdana), chosen by position rather than by result. The same four image IDs are shown at every level.\n\n![Level {L} comparison](comparisons/level_{L:02d}_comparison.jpg)\n\n![Level {L} close-up](comparisons/level_{L:02d}_closeup.jpg)\n")
    ids=[d['test_ids'][i] for i in (0,10,20,30)]; rows=[]
    for tid in ids:
        q=pm[(pm.level==L)&(pm.id==tid)].set_index('method')
        gt=q['gt'].dropna().iloc[0] if q['gt'].notna().any() else pd.read_csv(f'{TMP}/recomputed_L{L}.csv').set_index('id').loc[tid,'gt_text']
        for m in ORDER:
            if m not in q.index: continue
            r=q.loc[m]; o=r.ocr if isinstance(r.ocr,str) and r.ocr else ('(no text detected)' if L>0 else 'not run')
            rows.append([tid,m,f(r.psnr),f(r.ssim,4),f(r.char_acc,1,True),f(r.token_acc,0,True),f(r.cer,1,True),f'`{gt}`',f'`{o}`'])
    S.append("**Per-image metrics and OCR output for the four images shown**\n\n"+md(rows,['Image','Method','PSNR (dB)','SSIM','Char acc %','Token acc %','CER %','Ground-truth text','EasyOCR output'],['l','l','r','r','r','r','r','l','l']))
    S.append("\n**Observations**\n\n"+NOTES[L]+"\n")
    return '\n'.join(S)
