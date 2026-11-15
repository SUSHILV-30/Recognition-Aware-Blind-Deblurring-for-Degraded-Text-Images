import os, json, glob, numpy as np, pandas as pd, cv2, matplotlib, textwrap; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
H=os.path.expanduser('~'); OUT=f'{H}/mnt/Restromer-Level1-9/Restormer_Report_Levels0-9'; TMP=f'{H}/work/tmp'; FIG=f'{OUT}/figures'
os.makedirs(FIG,exist_ok=True); os.makedirs(f'{OUT}/comparisons',exist_ok=True)
ORDER=['Blurred input','Pretrained Restormer','Fine-tuned Restormer','Sharp GT (upper bound)']
COL={'Blurred input':'#2a78d6','Pretrained Restormer':'#eb6834','Fine-tuned Restormer':'#1baf7a','Sharp GT (upper bound)':'#eda100'}
MK={'Blurred input':'o','Pretrained Restormer':'s','Fine-tuned Restormer':'D','Sharp GT (upper bound)':'^'}
INK='#0b0b0b'; SEC='#52514e'
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8b8a85','axes.grid':True,'grid.color':'#e4e3df','grid.linewidth':0.8,'axes.axisbelow':True,'figure.facecolor':'white','axes.facecolor':'white','axes.titleweight':'bold','axes.titlesize':11.5,'text.color':INK,'axes.labelcolor':SEC,'xtick.color':SEC,'ytick.color':SEC})
pm=pd.read_csv(f'{OUT}/tables/all_levels_per_image_metrics.csv'); agg=pd.read_csv(f'{OUT}/tables/all_levels_summary_extended.csv'); gains=pd.read_csv(f'{OUT}/tables/gains_fine_tuned.csv'); tr=pd.read_csv(f'{OUT}/tables/training_and_registration_by_level.csv'); bf=pd.read_csv(f'{OUT}/tables/all_levels_by_font.csv')
info=json.load(open(f'{TMP}/info.json'))
which=os.environ.get('WHICH','all')
# ---------- comparison grids ----------
if which in('all','cmp'):
  for L in range(10):
    ids=[info[str(L)]['test_ids'][i] for i in (0,10,20,30)]
    fig,axes=plt.subplots(4,3,figsize=(17,16.6)); 
    for r,tid in enumerate(ids):
        q=pm[(pm.level==L)&(pm.id==tid)].set_index('method')
        gt_text=q['gt'].dropna().iloc[0] if q['gt'].notna().any() else pd.read_csv(f'{TMP}/recomputed_L{L}.csv').set_index('id').loc[tid,'gt_text']
        for c,(key,mname,head) in enumerate([('blurred',ORDER[0],'Blurred input (CAM02)'),('finetuned',ORDER[2],'Fine-tuned Restormer'),('groundtruth',ORDER[3],'Ground truth (CAM01, registered)')]):
            ax=axes[r,c]; im=cv2.imread(f'{OUT}/images/level_{L:02d}/L{L}_{tid}_{key}.jpg',0)
            ax.imshow(im,cmap='gray',vmin=0,vmax=255); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values(): s.set_visible(True); s.set_color('#8b8a85')
            m=q.loc[mname] if mname in q.index else pd.Series({'psnr':np.nan,'ssim':np.nan,'char_acc':np.nan,'token_acc':np.nan,'ocr':np.nan}); t=head if r==0 else ''
            if c<2: sub=f"PSNR {m.psnr:.2f} dB   ·   SSIM {m.ssim:.4f}"
            else: sub='reference (PSNR / SSIM are measured against this image)'
            ax.set_title((t+'\n' if t else '')+sub,fontsize=11.5,loc='center',fontweight='bold' if t else 'normal')
            if L>0:
                o=m.ocr if isinstance(m.ocr,str) and len(m.ocr)>0 else '(no text detected)'
                ax.set_xlabel(f'EasyOCR  [char acc {100*m.char_acc:.1f}% · token acc {100*m.token_acc:.0f}%]\n'+o,fontsize=10.5,family='monospace',color=INK)
            if c==0: ax.set_ylabel(f"{tid}\nGT text: {gt_text}",fontsize=9.5,color=INK)
    sg=info[str(L)].get('sigma') or info[str(L)].get('sigma_log') or 25
    fig.suptitle(f'Level {L} — blurred input | fine-tuned Restormer output | ground truth   (4 test images: 2 Times, 2 Verdana; estimated blur σ ≈ {sg} px)'+('\nOCR was not run for Level 0 in the exported results' if L==0 else ''),fontsize=14.5,fontweight='bold',y=0.995)
    plt.tight_layout(rect=(0,0,1,0.975)); plt.savefig(f'{OUT}/comparisons/level_{L:02d}_comparison.jpg',dpi=105,pil_kwargs={'quality':88}); plt.close()
    # close-up crops
    fig,axes=plt.subplots(4,3,figsize=(17,9.2))
    for r,tid in enumerate(ids):
        for c,(key,head) in enumerate([('blurred','Blurred input'),('finetuned','Fine-tuned Restormer'),('groundtruth','Ground truth')]):
            im=cv2.imread(f'{OUT}/images/level_{L:02d}/L{L}_{tid}_{key}.jpg',0); Hh,W=im.shape
            ax=axes[r,c]; ax.imshow(im[Hh//2-150:Hh//2+150, W//2-500:W//2+500],cmap='gray',vmin=0,vmax=255); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r==0: ax.set_title(head,fontsize=12)
            if c==0: ax.set_ylabel(tid,fontsize=9)
    fig.suptitle(f'Level {L} — centre crops at native resolution (1000 × 300 px)',fontsize=13.5,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/comparisons/level_{L:02d}_closeup.jpg',dpi=105,pil_kwargs={'quality':90}); plt.close()
    print('cmp',L,flush=True)
if which in('all','charts'):
    # A trends
    fig,axs=plt.subplots(2,2,figsize=(14,9))
    for ax,(met,ttl,sc) in zip(axs.ravel(),[('psnr','PSNR (dB) — higher is better',1),('ssim','SSIM — higher is better',1),('char_acc','EasyOCR character accuracy (%)',100),('token_acc','EasyOCR token (word) accuracy (%)',100)]):
        for m in ORDER:
            s=agg[agg.method==m].sort_values('level'); s=s[s[met].notna()]
            if len(s)==0: continue
            ax.plot(s.level,s[met]*sc,color=COL[m],marker=MK[m],ms=7,lw=2,label=m,markeredgecolor='white',markeredgewidth=1)
        ax.set_title(ttl,loc='left'); ax.set_xticks(range(10)); ax.set_xlabel('blur level (HDC 2021 focus step)')
    axs[0,0].legend(frameon=False,fontsize=9.5); axs[1,0].legend(frameon=False,fontsize=9.5,loc='center left')
    s=agg[agg.method==ORDER[2]].set_index('level')
    for L in (1,5,9):
        axs[0,0].annotate(f"{s.loc[L,'psnr']:.1f}",(L,s.loc[L,'psnr']),textcoords='offset points',xytext=(0,8),ha='center',fontsize=9,color=INK)
        if L>1: axs[1,0].annotate(f"{100*s.loc[L,'char_acc']:.1f}",(L,100*s.loc[L,'char_acc']),textcoords='offset points',xytext=(0,8),ha='center',fontsize=9,color=INK)
    fig.suptitle('Test-set means by blur level (n = 40 images per level; Level 0 has PSNR/SSIM only)',fontsize=13,fontweight='bold'); plt.tight_layout(); plt.savefig(f'{FIG}/trends_all_levels.png',dpi=150); plt.close()
    # B gains
    fig,axs=plt.subplots(1,3,figsize=(16,4.6)); g=gains
    for ax,(col,ttl) in zip(axs,[('dpsnr_vs_blur','PSNR gain over blurred input (dB)'),('dssim_vs_blur','SSIM gain over blurred input'),('dchar_vs_blur_pts','Character-accuracy gain over blurred input (points)')]):
        gg=g[g[col].notna()]; b=ax.bar(gg.level,gg[col],color='#2a78d6',width=0.62)
        for x,v in zip(gg.level,gg[col]): ax.text(x,v+(0.02*abs(gg[col]).max() if v>=0 else -0.07*abs(gg[col]).max()),f'{v:+.2f}' if 'ssim' not in col and 'char' not in col else (f'{v:+.3f}' if 'ssim' in col else f'{v:+.1f}'),ha='center',fontsize=8,color=INK)
        if 'char' in col: ax.set_ylim(-16,108)
        ax.axhline(0,color=SEC,lw=1); ax.set_title(ttl,loc='left'); ax.set_xticks(range(10)); ax.set_xlabel('blur level'); ax.grid(axis='x',visible=False)
    plt.tight_layout(); plt.savefig(f'{FIG}/gains_fine_tuned_vs_blurred.png',dpi=150); plt.close()
    # C training curves
    ramp=['#86b6ef','#6da7ec','#5598e7','#3987e5','#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b']
    fig,axs=plt.subplots(1,3,figsize=(18,5.6))
    for L in range(10):
        h=info[str(L)].get('hist')
        if not h: continue
        axs[0].plot(h['iter'],h['loss'],color=ramp[L],lw=1.5); axs[0].text(h['iter'][-1]+30,h['loss'][-1],f'L{L}',fontsize=8.5,va='center')
        axs[1].plot(h['val_iter'],h['val_psnr'],color=ramp[L],lw=1.8,marker='o',ms=4); axs[1].text(h['val_iter'][-1]+30,h['val_psnr'][-1],f'L{L}',fontsize=8.5,va='center')
        axs[2].plot(h['val_iter'],h['val_ssim'],color=ramp[L],lw=1.8,marker='o',ms=4,label=f'Level {L}'); axs[2].text(h['val_iter'][-1]+30,h['val_ssim'][-1],f'L{L}',fontsize=8.5,va='center')
    axs[0].set_title('Training Charbonnier loss (mean of each 50 iterations)',loc='left'); axs[1].set_title('Validation PSNR (dB), 10 images',loc='left'); axs[2].set_title('Validation SSIM, 10 images',loc='left')
    for a in axs: a.set_xlabel('iteration'); a.set_xlim(0,3250)
    axs[2].legend(frameon=False,fontsize=8.5,ncol=2,loc='lower right',title='lighter = less blur')
    plt.tight_layout(); plt.savefig(f'{FIG}/training_curves_all_levels.png',dpi=150); plt.close()
    # D registration
    fig,axs=plt.subplots(1,2,figsize=(14,4.6)); t=tr.copy(); t.loc[t.level==5,'blur_sigma_px']=25
    axs[0].bar(t.level,t.blur_sigma_px,color='#2a78d6',width=0.62)
    for x,v in zip(t.level,t.blur_sigma_px): axs[0].text(x,v+0.8,f'{int(v)}',ha='center',fontsize=9.5)
    axs[0].set_title('Estimated Gaussian-equivalent blur σ (px)  — search grid 5…100 step 5',loc='left'); axs[0].set_xticks(range(10)); axs[0].set_xlabel('blur level'); axs[0].grid(axis='x',visible=False)
    axs[1].plot(t.level,t.ncc_before,color='#2a78d6',marker='o',lw=2,label='before registration',markeredgecolor='white'); axs[1].plot(t.level,t.ncc_after,color='#eb6834',marker='s',lw=2,label='after global affine',markeredgecolor='white'); axs[1].plot(t.level,t.ncc_after_min,color='#eb6834',lw=1.2,ls='--',label='after (worst pair)')
    axs[1].set_title('Alignment score: NCC(blurred-GT, real input) over text area, mean of 200 pairs',loc='left'); axs[1].set_xticks(range(10)); axs[1].set_xlabel('blur level'); axs[1].legend(frameon=False); axs[1].set_ylim(0.55,1.0)
    plt.tight_layout(); plt.savefig(f'{FIG}/registration_by_level.png',dpi=150); plt.close()
    # E fonts
    fig,axs=plt.subplots(1,2,figsize=(14,4.6)); f=bf[bf.method==ORDER[2]]
    for ax,(met,ttl,sc) in zip(axs,[('char_acc','Fine-tuned output: character accuracy by font (%)',100),('psnr','Fine-tuned output: PSNR by font (dB)',1)]):
        for font,c,mk in [('timesR','#2a78d6','o'),('verdanaRef','#eb6834','s')]:
            s=f[(f.font==font)&f[met].notna()].sort_values('level'); ax.plot(s.level,s[met]*sc,color=c,marker=mk,lw=2,ms=7,markeredgecolor='white',label={'timesR':'Times (serif)','verdanaRef':'Verdana (sans-serif)'}[font])
        ax.set_title(ttl,loc='left'); ax.set_xticks(range(10)); ax.set_xlabel('blur level'); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{FIG}/font_breakdown.png',dpi=150); plt.close()
    # G distribution
    fig,ax=plt.subplots(figsize=(11,4.6)); d=[100*pm[(pm.level==L)&(pm.method==ORDER[2])].char_acc.values for L in range(1,10)]
    bp=ax.boxplot(d,positions=range(1,10),widths=0.5,patch_artist=True,medianprops=dict(color=INK,lw=1.6),boxprops=dict(facecolor='#cde2fb',edgecolor='#1c5cab'),whiskerprops=dict(color='#1c5cab'),capprops=dict(color='#1c5cab'),flierprops=dict(marker='o',ms=4,markerfacecolor='#1c5cab',markeredgecolor='white'))
    ax.set_title('Per-image spread of fine-tuned character accuracy (%), 40 test images per level',loc='left'); ax.set_xlabel('blur level'); ax.grid(axis='x',visible=False)
    plt.tight_layout(); plt.savefig(f'{FIG}/char_acc_distribution.png',dpi=150); plt.close()
    # F PSF
    fig,axs=plt.subplots(2,10,figsize=(20,4.6))
    for L in range(10):
        p=np.load(f'{TMP}/psf_L{L}.npy').astype(np.float32)
        sm=cv2.GaussianBlur(p[0],(0,0),15); y,x=np.unravel_index(np.argmin(sm),sm.shape); y=int(np.clip(y,200,p.shape[1]-200)); x=int(np.clip(x,200,p.shape[2]-200))
        for r in range(2):
            a=axs[r,L]; c=p[r][y-200:y+200,x-200:x+200]; a.imshow(c,cmap='gray',vmin=float(c.min()),vmax=float(c.max())); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            if r==0: a.set_title(f'Level {L}',fontsize=11)
        if L==0: axs[0,0].set_ylabel('CAM01 (sharp)',fontsize=10); axs[1,0].set_ylabel('CAM02 (defocused)',fontsize=10)
    fig.suptitle('Point-target (PSF) images shipped with each HDC 2021 step — 400 × 400 px crop around the dot, contrast stretched per panel',fontsize=12.5,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{FIG}/psf_by_level.png',dpi=130); plt.close()
    # H pipeline
    fig,ax=plt.subplots(figsize=(20,4.4)); ax.set_xlim(0,20); ax.set_ylim(0,4.4); ax.axis('off')
    steps=[('1  Data','HDC 2021 step N\n200 pairs (100 Times,\n100 Verdana)\nCAM01 = sharp GT\nCAM02 = blurred input\nTIFF, 2360×1460'),('2  Normalise','mean → grayscale\n÷ 99.5th percentile\nclip to [0,1]'),('3  Split','seed 42, per font\n150 train / 10 val\n/ 40 test (disjoint)'),('4  Register','estimate blur σ (NCC)\npatch matching + RANSAC\none global affine\nwarp GT, trim 48 px'),('5  Fine-tune','Restormer, encoder frozen\nrandom text patches\nCharbonnier + AdamW\ncosine LR, AMP, 3000 it'),('6  Restore','best-val-SSIM ckpt\n768 px tiles, overlap 192\nfeathered blending'),('7  Evaluate','PSNR, SSIM vs GT\nEasyOCR → char / token\naccuracy vs GT text')]
    w=2.55
    for i,(t,b) in enumerate(steps):
        x=0.2+i*2.84
        ax.add_patch(FancyBboxPatch((x,0.5),w,3.2,boxstyle='round,pad=0.02,rounding_size=0.12',fc='#cde2fb' if i==4 else '#fcfcfb',ec='#1c5cab' if i==4 else SEC,lw=1.5))
        ax.text(x+w/2,3.35,t,ha='center',va='center',fontsize=12,fontweight='bold'); ax.text(x+w/2,1.85,b,ha='center',va='center',fontsize=9.6,linespacing=1.45)
        if i<6: ax.add_patch(FancyArrowPatch((x+w+0.02,2.1),(x+2.84-0.02,2.1),arrowstyle='-|>',mutation_scale=14,lw=1.6,color=INK))
    ax.text(10,4.15,'Per-level pipeline implemented by the training notebook (identical for every level; only LEVEL changes)',ha='center',fontsize=13,fontweight='bold')
    plt.savefig(f'{FIG}/pipeline_overview.png',dpi=140,bbox_inches='tight'); plt.close()
    print('charts ok')
