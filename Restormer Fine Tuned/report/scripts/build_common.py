import os, json, numpy as np, pandas as pd
H=os.path.expanduser('~'); OUT=f'{H}/mnt/Restromer-Level1-9/Restormer_Report_Levels0-9'; TMP=f'{H}/work/tmp'
ORDER=['Blurred input','Pretrained Restormer','Fine-tuned Restormer','Sharp GT (upper bound)']
pm=pd.read_csv(f'{OUT}/tables/all_levels_per_image_metrics.csv'); agg=pd.read_csv(f'{OUT}/tables/all_levels_summary_extended.csv'); gains=pd.read_csv(f'{OUT}/tables/gains_fine_tuned.csv').set_index('level'); tr=pd.read_csv(f'{OUT}/tables/training_and_registration_by_level.csv').set_index('level'); bf=pd.read_csv(f'{OUT}/tables/all_levels_by_font.csv')
info=json.load(open(f'{TMP}/info.json')); P=json.load(open(f'{TMP}/params.json'))
tr.loc[5,'blur_sigma_px']=25; tr.loc[5,'patch']=128; tr.loc[5,'batch']=4
def md(rows,head,align=None):
    align=align or ['l']+['r']*(len(head)-1)
    s='| '+' | '.join(head)+' |\n|'+'|'.join({'l':':--','r':'--:','c':':-:'}[a] for a in align)+'|\n'
    for r in rows: s+='| '+' | '.join(str(x) for x in r)+' |\n'
    return s
def f(x,n=2,pct=False,sign=False):
    if x is None or (isinstance(x,float) and np.isnan(x)): return '—'
    if pct: x=100*x
    return (f'{x:+.{n}f}' if sign else f'{x:.{n}f}')
def A(L,m): 
    q=agg[(agg.level==L)&(agg.method==m)]
    return q.iloc[0] if len(q) else None
def summary_table(levels=range(10)):
    rows=[]
    for L in levels:
        for m in ORDER:
            a=A(L,m)
            if a is None: continue
            rows.append([L,m,f(a.psnr),f(a.ssim,4),f(a.char_acc,2,True),f(a.token_acc,2,True),f(a.cer,2,True),f(a.wer,2,True),f(a.exact_match,1,True),f(a.empty_ocr,0,True)])
    return md(rows,['Level','Method','PSNR (dB)','SSIM','Char acc %','Token acc %','CER %','WER %','Exact-match %','Empty OCR %'],['r','l']+['r']*8)
def headline_table():
    rows=[]
    for L in range(10):
        b,p,ft,g=A(L,ORDER[0]),A(L,ORDER[1]),A(L,ORDER[2]),A(L,ORDER[3])
        rows.append([L,int(tr.loc[L,'blur_sigma_px']),f(b.psnr),f(ft.psnr),f(ft.psnr-b.psnr,2,sign=True),f(b.ssim,4),f(ft.ssim,4),f(ft.ssim-b.ssim,3,sign=True),f(b.char_acc,1,True),f(ft.char_acc,1,True),f(g.char_acc,1,True) if g is not None else '—',f(ft.char_acc-b.char_acc,1,True,True) if L>0 else '—'])
    return md(rows,['Level','σ est. (px)','PSNR blurred','PSNR fine-tuned','Δ PSNR','SSIM blurred','SSIM fine-tuned','Δ SSIM','Char acc blurred %','Char acc fine-tuned %','Char acc sharp GT %','Δ char acc (pts)'],['r']*12)
def gains_table():
    rows=[]
    for L in range(10):
        g=gains.loc[L]
        rows.append([L,f(g.dpsnr_vs_blur,2,sign=True),f(g.dssim_vs_blur,4,sign=True),f(g.dchar_vs_blur_pts,1,sign=True),f(g.dtoken_vs_blur_pts,1,sign=True),f(g.dpsnr_vs_pre,2,sign=True),f(g.dssim_vs_pre,4,sign=True),f(g.dchar_vs_pre_pts,1,sign=True),f'{int(g.psnr_wins)}/40',f'{int(g.ssim_wins)}/40',(f'{int(g.char_wins)} / {int(g.char_ties)} / {int(g.char_losses)}' if L>0 else '—'),f(g.ft_char_minus_gt_pts,1,sign=True)])
    return md(rows,['Level','Δ PSNR vs blurred','Δ SSIM vs blurred','Δ char acc vs blurred (pts)','Δ token acc vs blurred (pts)','Δ PSNR vs pretrained','Δ SSIM vs pretrained','Δ char acc vs pretrained (pts)','Images with PSNR gain','Images with SSIM gain','Char acc win / tie / loss vs blurred','Fine-tuned − sharp-GT char acc (pts)'],['r']*12)
def train_table():
    rows=[]
    for L in range(10):
        t=tr.loc[L]
        rows.append([L,t.date if isinstance(t.date,str) else '2026-10 (legacy run)',int(t.blur_sigma_px),f'{int(t.patch)}×{int(t.patch)} × {int(t.batch)}',int(3000*t.batch),f(t.val_psnr_start),f(t.val_ssim_start,4),int(t.best_iter),f(t.best_val_psnr),f(t.best_val_ssim,4),f(t.loss_first50,4),f(t.loss_last50,4),f(t.train_minutes,0) if not np.isnan(t.train_minutes) else '—'])
    return md(rows,['Level','Run date','σ est. (px)','Patch × batch','Patches seen','Val PSNR @0','Val SSIM @0','Best iter','Val PSNR @best','Val SSIM @best','Loss (first 50 it)','Loss (last 50 it)','Wall time (min)'],['r']*13)
def reg_table():
    rows=[]
    for L in range(10):
        t=tr.loc[L]; rows.append([L,int(t.blur_sigma_px),f(t.ncc_before,3),f(t.ncc_after,3),f(t.ncc_after_min,3),f(t.affine_scale_x,4),f(t.affine_scale_y,4),f(t.affine_tx,1),f(t.affine_ty,1)])
    return md(rows,['Level','σ est. (px)','NCC before','NCC after','NCC after (worst pair)','Affine scale x','Affine scale y','Shift x (px)','Shift y (px)'],['r']*9)
def font_table(levels=range(10)):
    rows=[]
    for L in levels:
        for m in ORDER:
            for fo in ['timesR','verdanaRef']:
                q=bf[(bf.level==L)&(bf.method==m)&(bf.font==fo)]
                if len(q)==0: continue
                q=q.iloc[0]; rows.append([L,m,fo,f(q.psnr),f(q.ssim,4),f(q.char_acc,2,True),f(q.token_acc,2,True),f(q.cer,2,True),f(q.wer,2,True)])
    return md(rows,['Level','Method','Font','PSNR (dB)','SSIM','Char acc %','Token acc %','CER %','WER %'],['r','l','l']+['r']*6)
def param_table():
    order=['patch_embed','encoder_level1','down1_2','encoder_level2','down2_3','encoder_level3','down3_4','latent','up4_3','reduce_chan_level3','decoder_level3','up3_2','reduce_chan_level2','decoder_level2','up2_1','decoder_level1','refinement','output']
    desc={'patch_embed':'3×3 conv, 3→48','encoder_level1':'4 blocks, C=48, 1 head, full res','down1_2':'3×3 conv 48→24 + PixelUnshuffle(2) → 96 ch','encoder_level2':'6 blocks, C=96, 2 heads, 1/2 res','down2_3':'3×3 conv 96→48 + PixelUnshuffle(2) → 192 ch','encoder_level3':'6 blocks, C=192, 4 heads, 1/4 res','down3_4':'3×3 conv 192→96 + PixelUnshuffle(2) → 384 ch','latent':'8 blocks, C=384, 8 heads, 1/8 res','up4_3':'3×3 conv 384→768 + PixelShuffle(2) → 192 ch','reduce_chan_level3':'1×1 conv 384→192 after concat','decoder_level3':'6 blocks, C=192, 4 heads, 1/4 res','up3_2':'3×3 conv 192→384 + PixelShuffle(2) → 96 ch','reduce_chan_level2':'1×1 conv 192→96 after concat','decoder_level2':'6 blocks, C=96, 2 heads, 1/2 res','up2_1':'3×3 conv 96→192 + PixelShuffle(2) → 48 ch','decoder_level1':'4 blocks, C=96 (48+48 concat), 1 head, full res','refinement':'4 blocks, C=96, 1 head, full res','output':'3×3 conv 96→3, then + input'}
    rows=[[f'`{k}`',desc[k],f"{P['modules'][k]:,}",f"{100*P['modules'][k]/P['total']:.2f}",'Frozen' if k in P['frozen_names'] else 'Trainable'] for k in order]
    rows.append(['**Total**','',f"**{P['total']:,}**",'100.00',f"Frozen {P['frozen']:,} · Trainable {P['trainable']:,}"])
    return md(rows,['Module (name in code)','What it is','Parameters','% of model','Fine-tuning status'],['l','l','r','r','l'])
