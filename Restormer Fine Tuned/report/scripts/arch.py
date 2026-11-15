import os, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
H=os.path.expanduser('~'); OUT=f'{H}/mnt/Restromer-Level1-9/Restormer_Report_Levels0-9/figures'; os.makedirs(OUT,exist_ok=True)
# ---------- analytic parameter count ----------
def tb(dim,heads,exp=2.66):
    h=int(dim*exp)
    attn=heads + dim*3*dim + 3*dim*9 + dim*dim
    ffn=dim*2*h + 2*h*9 + h*dim
    return 2*dim + attn + 2*dim + ffn
P={}
P['patch_embed']=3*48*9
P['encoder_level1']=4*tb(48,1); P['down1_2']=48*24*9
P['encoder_level2']=6*tb(96,2); P['down2_3']=96*48*9
P['encoder_level3']=6*tb(192,4); P['down3_4']=192*96*9
P['latent']=8*tb(384,8)
P['up4_3']=384*768*9; P['reduce_chan_level3']=384*192
P['decoder_level3']=6*tb(192,4)
P['up3_2']=192*384*9; P['reduce_chan_level2']=192*96
P['decoder_level2']=6*tb(96,2)
P['up2_1']=96*192*9
P['decoder_level1']=4*tb(96,1)
P['refinement']=4*tb(96,1)
P['output']=96*3*9
FRZ=['patch_embed','encoder_level1','down1_2','encoder_level2','down2_3','encoder_level3','down3_4']
tot=sum(P.values()); frz=sum(P[k] for k in FRZ)
import json
json.dump({'modules':P,'total':tot,'frozen':frz,'trainable':tot-frz,'frozen_names':FRZ},open(f'{H}/work/tmp/params.json','w'),indent=1)
print('total',tot,'frozen',frz,'trainable',tot-frz)
def M(k): return f'{P[k]/1e6:.2f}M' if P[k]>=1e4 else f'{P[k]/1e3:.1f}K'
INK='#0b0b0b'; SEC='#52514e'; FRZC='#e4e3df'; FRZE='#8b8a85'; TRC='#cde2fb'; TRE='#1c5cab'; OPC='#fcfcfb'
fig,ax=plt.subplots(figsize=(21,11.5)); ax.set_xlim(0,21); ax.set_ylim(0,11.5); ax.axis('off'); fig.patch.set_facecolor('white')
def box(x,y,w,h,title,lines,frozen=None,fs=10.5):
    fc=OPC if frozen is None else (FRZC if frozen else TRC); ec=SEC if frozen is None else (FRZE if frozen else TRE)
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=fc,ec=ec,lw=1.6,zorder=3))
    ax.text(x,y+h/2-0.27,title,ha='center',va='center',fontsize=fs,fontweight='bold',color=INK,zorder=4)
    ax.text(x,y-0.17,lines,ha='center',va='center',fontsize=fs-1.7,color=INK,zorder=4,linespacing=1.35)
def arr(p,q,style='-',color=INK,rad=0,lw=1.6):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=14,lw=lw,color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}',zorder=2))
def circ(x,y,t):
    ax.add_patch(Circle((x,y),0.22,fc='white',ec=INK,lw=1.5,zorder=5)); ax.text(x,y,t,ha='center',va='center',fontsize=11,fontweight='bold',zorder=6)
Y1,Y2,Y3,Y4=8.95,6.3,4.0,1.5; XE=4.6; XD=13.4; BW=3.3; BH=1.25
# top row
box(0.85,Y1,1.3,1.0,'Input',"blurred I\nH×W×3",None)
box(2.45,Y1,1.3,1.0,'Patch embed',f"3×3 conv\n3→48  ({M('patch_embed')})",True,fs=9.5)
box(XE,Y1,BW*0.72,BH,'Encoder L1  ❄',f"4 × Transformer block\nC=48 · 1 head · H×W\n{M('encoder_level1')}",True)
box(XE,Y2,BW*0.85,BH,'Encoder L2  ❄',f"6 × Transformer block\nC=96 · 2 heads · H/2×W/2\n{M('encoder_level2')}",True)
box(XE,Y3,BW,BH,'Encoder L3  ❄',f"6 × Transformer block\nC=192 · 4 heads · H/4×W/4\n{M('encoder_level3')}",True)
box(9.0,Y4,4.6,BH,'Latent (bottleneck)',f"8 × Transformer block · C=384 · 8 heads · H/8×W/8\n{M('latent')}  (55% of the network)",False)
box(XD,Y3,BW,BH,'Decoder L3',f"6 × Transformer block\nC=192 · 4 heads · H/4×W/4\n{M('decoder_level3')}",False)
box(XD,Y2,BW*0.85,BH,'Decoder L2',f"6 × Transformer block\nC=96 · 2 heads · H/2×W/2\n{M('decoder_level2')}",False)
box(XD,Y1,BW*0.72,BH,'Decoder L1',f"4 × Transformer block\nC=96 · 1 head · H×W\n{M('decoder_level1')}",False)
box(16.35,Y1,2.3,BH,'Refinement',f"4 × Transformer block\nC=96 · 1 head · H×W\n{M('refinement')}",False)
box(18.45,Y1,1.25,1.0,'Output conv',f"3×3 conv\n96→3  ({M('output')})",False,fs=9.5)
circ(19.6,Y1,'+')
box(20.4,Y1,0.95,1.0,'Output',"restored Î\nH×W×3",None,fs=9.5)
arr((1.5,Y1),(1.8,Y1)); arr((3.1,Y1),(XE-BW*0.36,Y1))
# down ops
def op(x,y,t,frozen): 
    ax.add_patch(FancyBboxPatch((x-1.05,y-0.27),2.1,0.54,boxstyle='round,pad=0.02,rounding_size=0.1',fc=FRZC if frozen else TRC,ec=FRZE if frozen else TRE,lw=1.2,zorder=3))
    ax.text(x,y,t,ha='center',va='center',fontsize=8.3,color=INK,zorder=4)
for (ya,yb,t) in [(Y1,Y2,f"↓ 3×3 conv 48→24 + PixelUnshuffle  ({M('down1_2')})"),(Y2,Y3,f"↓ 3×3 conv 96→48 + PixelUnshuffle  ({M('down2_3')})")]:
    ym=(ya+yb)/2; arr((XE,ya-BH/2),(XE,ym+0.27)); 
    ax.add_patch(FancyBboxPatch((XE-1.75,ym-0.27),3.5,0.54,boxstyle='round,pad=0.02,rounding_size=0.1',fc=FRZC,ec=FRZE,lw=1.2,zorder=3)); ax.text(XE,ym,t,ha='center',va='center',fontsize=8.3,zorder=4)
    arr((XE,ym-0.27),(XE,yb+BH/2))
# enc L3 -> latent
ax.add_patch(FancyBboxPatch((XE-1.85,2.48),3.7,0.54,boxstyle='round,pad=0.02,rounding_size=0.1',fc=FRZC,ec=FRZE,lw=1.2,zorder=3)); ax.text(XE,2.75,f"↓ 3×3 conv 192→96 + PixelUnshuffle  ({M('down3_4')})",ha='center',va='center',fontsize=8.3,zorder=4)
arr((XE,Y3-BH/2),(XE,3.02)); ax.plot([XE,XE],[2.48,Y4],color=INK,lw=1.6,zorder=2); arr((XE,Y4),(9.0-2.3,Y4))
# latent -> dec L3
ax.plot([9.0+2.3,XD],[Y4,Y4],color=INK,lw=1.6,zorder=2)
ax.add_patch(FancyBboxPatch((XD-1.85,2.02),3.7,0.54,boxstyle='round,pad=0.02,rounding_size=0.1',fc=TRC,ec=TRE,lw=1.2,zorder=3)); ax.text(XD,2.29,f"↑ 3×3 conv 384→768 + PixelShuffle  ({M('up4_3')})",ha='center',va='center',fontsize=8.3,zorder=4)
arr((XD,Y4),(XD,2.02)); 
def upjoin(yc_lo_top,ydec,uptxt,redtxt,ybase):
    pass
# concat + reduce before Dec L3
circ(XD,2.95,'C'); arr((XD,2.56),(XD,2.73))
ax.text(XD+0.35,2.95,f"concat → 1×1 conv 384→192 ({M('reduce_chan_level3')})",ha='left',va='center',fontsize=8.3,color=INK)
arr((XD,3.17),(XD,Y3-BH/2))
# dec L3 -> dec L2
ax.add_patch(FancyBboxPatch((XD-1.85,4.78),3.7,0.5,boxstyle='round,pad=0.02,rounding_size=0.1',fc=TRC,ec=TRE,lw=1.2,zorder=3)); ax.text(XD,5.03,f"↑ 3×3 conv 192→384 + PixelShuffle  ({M('up3_2')})",ha='center',va='center',fontsize=8.3,zorder=4)
arr((XD,Y3+BH/2),(XD,4.78)); circ(XD,5.48,'C'); ax.plot([XD,XD],[5.28,5.26],color=INK); arr((XD,5.70),(XD,Y2-BH/2),lw=1.4)
ax.text(XD+0.35,5.48,f"concat → 1×1 conv 192→96 ({M('reduce_chan_level2')})",ha='left',va='center',fontsize=8.3)
# dec L2 -> dec L1
ax.add_patch(FancyBboxPatch((XD-1.85,7.12),3.7,0.5,boxstyle='round,pad=0.02,rounding_size=0.1',fc=TRC,ec=TRE,lw=1.2,zorder=3)); ax.text(XD,7.37,f"↑ 3×3 conv 96→192 + PixelShuffle  ({M('up2_1')})",ha='center',va='center',fontsize=8.3,zorder=4)
arr((XD,Y2+BH/2),(XD,7.12)); circ(XD,7.97,'C'); ax.text(XD+0.35,7.98,"concat 48+48 = 96 (no 1×1 reduction)",ha='left',va='center',fontsize=8.3)
# skips
SK='#52514e'
arr((XE+BW*0.5,Y3),(XD-BW*0.5-0.9,Y3),style=(0,(5,3)),color=SK); ax.plot([XD-BW*0.5-0.9,XD-BW*0.5-0.9],[Y3,2.95],color=SK,lw=1.6,ls=(0,(5,3))); arr((XD-BW*0.5-0.9,2.95),(XD-0.22,2.95),style=(0,(5,3)),color=SK)
ax.text(9.0,Y3+0.18,'skip connection (encoder L3 features, 192 ch)',ha='center',fontsize=9,color=SK,style='italic')
arr((XE+BW*0.425,Y2),(XD-BW*0.5-0.9,Y2),style=(0,(5,3)),color=SK); ax.plot([XD-BW*0.5-0.9]*2,[Y2,5.48],color=SK,lw=1.6,ls=(0,(5,3))); arr((XD-BW*0.5-0.9,5.48),(XD-0.22,5.48),style=(0,(5,3)),color=SK)
ax.text(9.0,Y2+0.18,'skip connection (encoder L2 features, 96 ch)',ha='center',fontsize=9,color=SK,style='italic')
arr((XE+BW*0.36,Y1),(XD-BW*0.5-0.9,Y1),style=(0,(5,3)),color=SK); ax.plot([XD-BW*0.5-0.9]*2,[Y1,7.97],color=SK,lw=1.6,ls=(0,(5,3))); arr((XD-BW*0.5-0.9,7.97),(XD-0.22,7.97),style=(0,(5,3)),color=SK)
ax.text(8.0,Y1+0.18,'skip connection (encoder L1 features, 48 ch)',ha='center',fontsize=9,color=SK,style='italic')
arr((XD,8.19),(XD,Y1-BH/2))
arr((XD+BW*0.36,Y1),(16.35-1.15,Y1)); arr((17.5,Y1),(17.82,Y1)); arr((19.08,Y1),(19.38,Y1)); arr((19.82,Y1),(19.92,Y1))
# global residual
ax.plot([0.85,0.85],[Y1+0.5,10.4],color=INK,lw=1.6); ax.plot([0.85,19.6],[10.4,10.4],color=INK,lw=1.6); arr((19.6,10.4),(19.6,Y1+0.22))
ax.text(10.2,10.58,'global residual:  Î = I + R   (the network only predicts the correction R)',ha='center',fontsize=10.5,color=INK,style='italic')
ax.text(10.5,11.15,'Restormer (single-image defocus deblurring configuration) — as fine-tuned in this project',ha='center',fontsize=15,fontweight='bold')
# legend
lx,ly=16.0,4.6
ax.add_patch(FancyBboxPatch((lx,ly-0.2),0.5,0.4,boxstyle='round,pad=0.02',fc=FRZC,ec=FRZE,lw=1.4)); ax.text(lx+0.7,ly,f'❄ Frozen during fine-tuning — {frz/1e6:.2f}M params',va='center',fontsize=10)
ax.add_patch(FancyBboxPatch((lx,ly-0.8),0.5,0.4,boxstyle='round,pad=0.02',fc=TRC,ec=TRE,lw=1.4)); ax.text(lx+0.7,ly-0.6,f'Trainable — {(tot-frz)/1e6:.2f}M params',va='center',fontsize=10)
ax.text(0.3,1.25,f'Total {tot/1e6:.2f}M params · all convs bias-free\nTransformer block = LN → MDTA → + → LN → GDFN → +\nFFN expansion γ = 2.66 · LayerNorm type WithBias\nC = concatenate along channels',va='top',fontsize=9.3,color=SEC,linespacing=1.5)
plt.savefig(f'{OUT}/restormer_architecture.png',dpi=140,bbox_inches='tight',facecolor='white'); plt.savefig(f'{OUT}/restormer_architecture.svg',bbox_inches='tight',facecolor='white'); plt.close()

# ---------- Transformer block detail ----------
fig,axs=plt.subplots(3,1,figsize=(19,11.5),gridspec_kw={'height_ratios':[0.8,1.25,1]}); fig.patch.set_facecolor('white')
def flow(ax,items,y=0.5,title='',x0=0.02,gap=0.012):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.text(0.0,0.97,title,fontsize=13.5,fontweight='bold',va='top')
    x=x0; cent=[]
    for (t,w,kind) in items:
        if kind=='c':
            ax.add_patch(matplotlib.patches.Ellipse((x+w/2,y),w*0.8,0.2,fc='white',ec=INK,lw=1.4)); ax.text(x+w/2,y,t,ha='center',va='center',fontsize=12,fontweight='bold')
        else:
            fc={'a':TRC,'n':'#fcfcfb','g':'#e4e3df'}[kind]; ec={'a':TRE,'n':SEC,'g':FRZE}[kind]
            ax.add_patch(FancyBboxPatch((x,y-0.2),w,0.4,boxstyle='round,pad=0.004,rounding_size=0.012',fc=fc,ec=ec,lw=1.4)); ax.text(x+w/2,y,t,ha='center',va='center',fontsize=9.6,linespacing=1.3)
        cent.append((x,x+w)); x+=w+gap+0.022
    for i in range(len(cent)-1):
        ax.add_patch(FancyArrowPatch((cent[i][1]+0.002,y),(cent[i+1][0]-0.002,y),arrowstyle='-|>',mutation_scale=12,lw=1.4,color=INK))
    return cent
c=flow(axs[0],[('X\n(C×H×W)',0.07,'n'),('LayerNorm\n(over channels,\nper pixel)',0.1,'g'),('MDTA\nmulti-Dconv head\ntransposed attention',0.15,'a'),('+',0.035,'c'),('LayerNorm',0.08,'g'),('GDFN\ngated-Dconv\nfeed-forward',0.13,'a'),('+',0.035,'c'),('X_out\n(C×H×W)',0.07,'n')],title='(a) Transformer block:   X′ = X + MDTA(LN(X));   X_out = X′ + GDFN(LN(X′))')
a=axs[0]
for (src,dst) in [(0,3),(3,6)]:
    xs=(c[src][0]+c[src][1])/2 if src==0 else (c[src][0]+c[src][1])/2; xd=(c[dst][0]+c[dst][1])/2
    a.plot([xs,xs],[0.5+ (0.2 if src==0 else 0.1),0.84],color=SEC,lw=1.3,ls='--'); a.plot([xs,xd],[0.84,0.84],color=SEC,lw=1.3,ls='--'); a.add_patch(FancyArrowPatch((xd,0.84),(xd,0.6),arrowstyle='-|>',mutation_scale=11,lw=1.3,color=SEC,linestyle='--'))
a.text((c[0][0]+c[3][1])/2,0.87,'identity skip',ha='center',fontsize=9,color=SEC,style='italic'); a.text((c[3][0]+c[6][1])/2,0.87,'identity skip',ha='center',fontsize=9,color=SEC,style='italic')
flow(axs[1],[('LN(X)\nC×H×W',0.07,'n'),('1×1 conv\nC → 3C\n(pixel-wise mixing)',0.105,'a'),('3×3 depth-wise\nconv (3C groups)\nlocal context',0.105,'a'),('split → Q, K, V\nreshape each to\nheads × (C/h) × HW',0.12,'n'),('L2-normalise\nQ and K\nalong HW',0.09,'g'),('A = softmax(τ · Q Kᵀ)\nsize (C/h)×(C/h)\nτ learnable per head',0.14,'a'),('A · V\nreshape back\nto C×H×W',0.085,'n'),('1×1 conv\nC → C',0.065,'a')],y=0.52,gap=-0.004,title='(b) MDTA — attention is computed between CHANNELS (C/h × C/h map), not between pixels (HW × HW)')
axs[1].text(0.0,0.14,'Why it matters: a pixel-to-pixel attention map on a 768×768 tile would need (589,824)² entries; the transposed (channel) map needs only (C/h)² entries per head (48² or 96² here), so cost grows linearly with image size.\nGlobal context still enters because every entry of Q Kᵀ is a sum over ALL HW positions. The depth-wise conv before the split injects local (3×3) structure — strokes and edges — into Q, K and V.',fontsize=9.8,color=SEC,va='center',linespacing=1.5)
flow(axs[2],[('LN(X′)\nC×H×W',0.075,'n'),('1×1 conv\nC → 2·⌊2.66C⌋\n(expand)',0.12,'a'),('3×3 depth-wise conv\n(spatial context\ninside the FFN)',0.13,'a'),('chunk into two\nhalves  x₁ , x₂\n(⌊2.66C⌋ ch each)',0.12,'n'),('GELU(x₁) ⊙ x₂\n(gating: x₁ decides\nwhat of x₂ passes)',0.14,'a'),('1×1 conv\n⌊2.66C⌋ → C',0.1,'a')],y=0.55,title='(c) GDFN — a gated feed-forward network with a depth-wise convolution')
axs[2].text(0.0,0.13,'Hidden widths used here: C=48 → 127,  C=96 → 255,  C=192 → 510,  C=384 → 1021.  The gate lets each level suppress uninformative features (flat paper) and pass informative ones (stroke edges).',fontsize=9.8,color=SEC,va='center')
plt.tight_layout(); plt.savefig(f'{OUT}/transformer_block_detail.png',dpi=140,bbox_inches='tight',facecolor='white'); plt.close()
print('arch ok')
