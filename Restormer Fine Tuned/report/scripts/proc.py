import os, sys, glob, json, re, zipfile, io, subprocess, shutil
import numpy as np, pandas as pd, cv2, tifffile
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
H=os.path.expanduser('~'); MNT=f'{H}/mnt/Restromer-Level1-9'; P2=f'{H}/work/p2/FinalYearProject-Phase2'
OUT=f'{MNT}/Restormer_Report_Levels0-9'; TMP=f'{H}/work/tmp'
levels=[int(a) for a in sys.argv[1:]] or list(range(10))
MARGIN=48
def normalise(img):
    bg=np.percentile(img,99.5)
    if bg<=0: bg=img.max()+1e-6
    return np.clip(img/bg,0,1)
def read_raw(b):
    img=tifffile.imread(io.BytesIO(b)).astype(np.float32)
    if img.ndim==3: img=img.mean(-1)
    return img
def warp_gt(s,M):
    Hh,W=s.shape
    return cv2.warpAffine(s,M,(W,Hh),flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
def met(p,g):
    p=p.astype(np.float32); g=g.astype(np.float32)
    return float(PSNR(g,p,data_range=1.0)), float(SSIM(g,p,data_range=1.0))
# index of restored pngs
ridx={}
for z in sorted(glob.glob(f'{MNT}/FinalYearProject-Phase2-*.zip')):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if '/restored/' in n and n.endswith('.png'): ridx[n]=z
def rest_png(L,kind,tid):
    cands=[n for n in ridx if n.startswith(f'FinalYearProject-Phase2/level_{L:02d}/') and n.endswith(f'/restored/{kind}/{tid}.png')]
    n=cands[0]
    with zipfile.ZipFile(ridx[n]) as zf: b=zf.read(n)
    im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
    return im.astype(np.float32)/65535.0
def u8(x): return (np.clip(x,0,1)*255).round().astype(np.uint8)
for L in levels:
    dz=[z for z in glob.glob(f'{MNT}/HDC2021_step*.zip') if re.search(rf'step_?{L}-',os.path.basename(z))][0]
    zf=zipfile.ZipFile(dz); names=zf.namelist()
    R=glob.glob(f'{P2}/level_{L:02d}/results/**/split.json',recursive=True)[0].rsplit('/',1)[0]
    split=json.load(open(f'{R}/split.json')); M=np.load(f'{R}/registration_affine.npy')
    test=split['test']; sel=[test[0],test[10],test[20],test[30]]
    todo=test if L==0 else sel
    a,b=[int(v) for v in os.environ.get('CHUNK','0:999').split(':')]; todo=todo[a:b]
    pi=None
    f=glob.glob(f'{R}/level{L}_per_image.csv')
    if f: pi=pd.read_csv(f[0])
    os.makedirs(f'{OUT}/images/level_{L:02d}',exist_ok=True)
    rows=[]
    for tid in todo:
        font,num=tid.split('_')
        def member(cam,ext):
            c=[n for n in names if '__MACOSX' not in n and f'/{cam}/' in n and re.search(rf'focusStep_{L}_{font}_size_\d+_sample_{num}\.{ext}$',n)]
            return c[0]
        s=normalise(read_raw(zf.read(member('CAM01','tif')))); b=normalise(read_raw(zf.read(member('CAM02','tif'))))
        gt_txt=zf.read(member('CAM01','txt')).decode('utf-8','ignore')
        w=np.clip(warp_gt(s,M),0,1)
        b_c=b[MARGIN:-MARGIN,MARGIN:-MARGIN].astype(np.float16); g_c=w[MARGIN:-MARGIN,MARGIN:-MARGIN].astype(np.float16)
        ft=rest_png(L,'finetuned',tid); pre=rest_png(L,'pretrained',tid)
        r={'level':L,'id':tid,'font':font,'gt_text':' '.join(gt_txt.split()),'shape':str(b_c.shape)}
        for k,im in [('blur',b_c),('pre',pre),('ft',ft)]:
            p,ss=met(im,g_c); r[f'{k}_psnr']=p; r[f'{k}_ssim']=ss
        if pi is not None:
            for k,mname in [('blur','Blurred input'),('pre','Pretrained Restormer'),('ft','Fine-tuned Restormer')]:
                q=pi[(pi.method==mname)&(pi.id==tid)].iloc[0]
                r[f'{k}_psnr_csv']=q.psnr; r[f'{k}_ssim_csv']=q.ssim
        rows.append(r)
        if tid in sel:
            for k,im in [('blurred',b_c),('pretrained',pre),('finetuned',ft),('groundtruth',g_c)]:
                cv2.imwrite(f'{OUT}/images/level_{L:02d}/L{L}_{tid}_{k}.jpg',u8(im.astype(np.float32)),[cv2.IMWRITE_JPEG_QUALITY,90])
        print(L,tid,{k:round(v,4) for k,v in r.items() if isinstance(v,float)},flush=True)
    pd.DataFrame(rows).to_csv(f"{TMP}/recomputed_L{L}{os.environ.get('SUF','')}.csv",index=False)
    # PSF
    try:
        c=[n for n in names if '__MACOSX' not in n and '/CAM02/' in n and n.endswith('_PSF.tif') and '/Times/' in n][0]
        psf=normalise(read_raw(zf.read(c)))
        c1=[n for n in names if '__MACOSX' not in n and '/CAM01/' in n and n.endswith('_PSF.tif') and '/Times/' in n][0]
        psf1=normalise(read_raw(zf.read(c1)))
        np.save(f'{TMP}/psf_L{L}.npy',np.stack([psf1,psf]).astype(np.float16)); print('psf',psf.shape,flush=True)
    except Exception as e: print('psf fail',e,flush=True)
print('DONE',flush=True)
