import os, glob, json, re, numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from datetime import datetime
H=os.path.expanduser('~'); P2=f'{H}/work/p2/FinalYearProject-Phase2'; OUT=f'{H}/mnt/Restromer-Level1-9/Restormer_Report_Levels0-9'; TMP=f'{H}/work/tmp'
os.makedirs(f'{OUT}/tables',exist_ok=True)
ORDER=['Blurred input','Pretrained Restormer','Fine-tuned Restormer','Sharp GT (upper bound)']
def clean(s): return ' '.join(str(s).split()) if isinstance(s,str) else ''
per=[]; ocr=[]; info={}
for L in range(10):
    R=glob.glob(f'{P2}/level_{L:02d}/results/**/split.json',recursive=True)[0].rsplit('/',1)[0]
    d={'level':L}
    mf=f'{P2}/level_{L:02d}/manifest.json'
    m=json.load(open(mf)) if os.path.exists(mf) else {}
    d['sigma']=m.get('blur_sigma_px'); d['cfg']=m.get('train_config'); d['best_val_ssim']=m.get('best_val_ssim')
    hf=glob.glob(f'{P2}/level_{L:02d}/**/training_history.json',recursive=True)
    if hf:
        h=json.load(open(hf[0])); d['hist']=h
        d['val0']=(h['val_psnr'][0],h['val_ssim'][0]); i=int(np.argmax(h['val_ssim'])); d['best']=(h['val_iter'][i],h['val_psnr'][i],h['val_ssim'][i]); d['val_last']=(h['val_psnr'][-1],h['val_ssim'][-1])
        d['loss_first']=h['loss'][0]; d['loss_last']=h['loss'][-1]; d['n_val']=len(h['val_iter'])
    lf=f'{P2}/level_{L:02d}/logs/train_log.txt'
    if os.path.exists(lf):
        t=open(lf).read(); d['log']=t
        ts=re.findall(r'\[(.*?)\]',t); a=datetime.strptime(ts[0],'%Y-%m-%d %H:%M:%S'); b=datetime.strptime(ts[-1],'%Y-%m-%d %H:%M:%S'); d['train_min']=round((b-a).total_seconds()/60,1); d['date']=ts[0][:10]
        mm=re.search(r'sigma (\d+)px \| patch (\d+)x(\d+)',t)
        if mm: d['sigma_log']=int(mm.group(1)); d['patch_log']=(int(mm.group(2)),int(mm.group(3)))
    c=pd.read_csv(f'{R}/pairing_checks.csv'); d['ncc']=(c.ncc_before.mean(),c.ncc_after.mean(),c.ncc_after.min())
    d['affine']=np.load(f'{R}/registration_affine.npy').tolist()
    s=json.load(open(f'{R}/split.json')); d['split']={k:len(v) for k,v in s.items()}; d['test_ids']=s['test']
    f=glob.glob(f'{R}/level{L}_per_image.csv')
    if f:
        p=pd.read_csv(f[0]); p.insert(0,'level',L); per.append(p)
        o=pd.read_csv(f'{R}/ocr_outputs.csv'); o.insert(0,'level',L); ocr.append(o)
    else:
        r=pd.read_csv(f'{TMP}/recomputed_L{L}.csv') if os.path.exists(f'{TMP}/recomputed_L{L}.csv') else None
        if r is not None:
            rows=[]
            for k,mn in [('blur',ORDER[0]),('pre',ORDER[1]),('ft',ORDER[2])]:
                for _,q in r.iterrows(): rows.append({'level':L,'method':mn,'id':q.id,'font':q.font,'psnr':q[f'{k}_psnr'],'ssim':q[f'{k}_ssim'],'char_acc':np.nan,'token_acc':np.nan})
            per.append(pd.DataFrame(rows)); d['recomputed']=True
    info[L]=d
per=pd.concat(per,ignore_index=True); ocr=pd.concat(ocr,ignore_index=True)
ocr['gt']=ocr['gt'].map(clean); ocr['ocr']=ocr['ocr'].map(clean)
def wer(p,g):
    g=g.split(); p=p.split()
    return Levenshtein.distance(p,g)/max(1,len(g))
ocr['edit_dist']=[Levenshtein.distance(p,g) for p,g in zip(ocr['ocr'],ocr['gt'])]
ocr['cer']=ocr['edit_dist']/ocr['gt'].str.len()
ocr['char_acc_re']=(1-ocr['cer']).clip(lower=0)
ocr['wer']=[wer(p,g) for p,g in zip(ocr['ocr'],ocr['gt'])]
ocr['exact']=(ocr['ocr']==ocr['gt']).astype(float); ocr['empty']=(ocr['ocr'].str.len()==0).astype(float)
ocr['nlev_sim']=[Levenshtein.normalized_similarity(p,g) for p,g in zip(ocr['ocr'],ocr['gt'])]
mrg=per.merge(ocr[['level','method','id','gt','ocr','edit_dist','cer','wer','exact','empty','nlev_sim','char_acc_re']],on=['level','method','id'],how='left')
chk=(mrg.char_acc-mrg.char_acc_re).abs().max(); print('char_acc recompute max diff',chk)
mrg.drop(columns=['char_acc_re']).to_csv(f'{OUT}/tables/all_levels_per_image_metrics.csv',index=False)
ocr.drop(columns=['char_acc_re']).to_csv(f'{OUT}/tables/all_levels_ocr_outputs.csv',index=False)
agg=mrg.groupby(['level','method']).agg(n=('id','count'),psnr=('psnr','mean'),psnr_std=('psnr','std'),psnr_min=('psnr','min'),psnr_max=('psnr','max'),ssim=('ssim','mean'),ssim_std=('ssim','std'),ssim_min=('ssim','min'),ssim_max=('ssim','max'),char_acc=('char_acc','mean'),char_acc_std=('char_acc','std'),char_acc_min=('char_acc','min'),char_acc_max=('char_acc','max'),token_acc=('token_acc','mean'),cer=('cer','mean'),wer=('wer','mean'),nlev_sim=('nlev_sim','mean'),edit_dist=('edit_dist','mean'),exact_match=('exact','mean'),empty_ocr=('empty','mean')).reset_index()
agg['method']=pd.Categorical(agg.method,ORDER,ordered=True); agg=agg.sort_values(['level','method']); agg.to_csv(f'{OUT}/tables/all_levels_summary_extended.csv',index=False)
bf=mrg.groupby(['level','method','font']).agg(psnr=('psnr','mean'),ssim=('ssim','mean'),char_acc=('char_acc','mean'),token_acc=('token_acc','mean'),cer=('cer','mean'),wer=('wer','mean')).reset_index()
bf['method']=pd.Categorical(bf.method,ORDER,ordered=True); bf=bf.sort_values(['level','method','font']); bf.to_csv(f'{OUT}/tables/all_levels_by_font.csv',index=False)
# gains + paired wins
g=[]
for L in range(10):
    a=agg[agg.level==L].set_index('method'); 
    if 'Fine-tuned Restormer' not in a.index: continue
    ft,bl,pr=a.loc[ORDER[2]],a.loc[ORDER[0]],a.loc[ORDER[1]]
    sub=mrg[mrg.level==L].pivot(index='id',columns='method',values=['psnr','ssim','char_acc'])
    row={'level':L,'dpsnr_vs_blur':ft.psnr-bl.psnr,'dssim_vs_blur':ft.ssim-bl.ssim,'dchar_vs_blur_pts':100*(ft.char_acc-bl.char_acc),'dtoken_vs_blur_pts':100*(ft.token_acc-bl.token_acc),'dpsnr_vs_pre':ft.psnr-pr.psnr,'dssim_vs_pre':ft.ssim-pr.ssim,'dchar_vs_pre_pts':100*(ft.char_acc-pr.char_acc),
         'psnr_wins':int((sub['psnr'][ORDER[2]]>sub['psnr'][ORDER[0]]).sum()),'ssim_wins':int((sub['ssim'][ORDER[2]]>sub['ssim'][ORDER[0]]).sum())}
    if L>0:
        row['char_wins']=int((sub['char_acc'][ORDER[2]]>sub['char_acc'][ORDER[0]]).sum()); row['char_ties']=int((sub['char_acc'][ORDER[2]]==sub['char_acc'][ORDER[0]]).sum()); row['char_losses']=int((sub['char_acc'][ORDER[2]]<sub['char_acc'][ORDER[0]]).sum())
        gt=a.loc[ORDER[3]]; row['ft_char_minus_gt_pts']=100*(ft.char_acc-gt.char_acc)
    g.append(row)
g=pd.DataFrame(g); g.to_csv(f'{OUT}/tables/gains_fine_tuned.csv',index=False)
tr=[]
for L,d in info.items():
    tr.append({'level':L,'date':d.get('date'),'blur_sigma_px':d.get('sigma') or d.get('sigma_log'),'patch':(d.get('cfg') or {}).get('patch') or (d.get('patch_log') or [None])[0],'batch':(d.get('cfg') or {}).get('batch') or (d.get('patch_log') or [None,None])[1],
     'ncc_before':d['ncc'][0],'ncc_after':d['ncc'][1],'ncc_after_min':d['ncc'][2],'val_psnr_start':d.get('val0',[None,None])[0],'val_ssim_start':d.get('val0',[None,None])[1],'best_iter':d.get('best',[None]*3)[0],'best_val_psnr':d.get('best',[None]*3)[1],'best_val_ssim':d.get('best',[None]*3)[2],'final_val_psnr':d.get('val_last',[None,None])[0],'final_val_ssim':d.get('val_last',[None,None])[1],'loss_first50':d.get('loss_first'),'loss_last50':d.get('loss_last'),'train_minutes':d.get('train_min'),'affine_scale_x':d['affine'][0][0],'affine_scale_y':d['affine'][1][1],'affine_tx':d['affine'][0][2],'affine_ty':d['affine'][1][2]})
tr=pd.DataFrame(tr); tr.to_csv(f'{OUT}/tables/training_and_registration_by_level.csv',index=False)
json.dump({str(k):{kk:vv for kk,vv in v.items() if kk not in('log',)} for k,v in info.items()},open(f'{TMP}/info.json','w'))
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(agg.round(4).to_string(index=False)); print(g.round(3).to_string(index=False)); print(tr.round(4).to_string(index=False)); print(bf[bf.method==ORDER[2]].round(4).to_string(index=False))
print(open(f'{P2}/level_05/results/level5_results/training_history.json').read()[:300])
