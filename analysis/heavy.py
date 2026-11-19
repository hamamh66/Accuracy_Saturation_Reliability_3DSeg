import os
OUT=os.environ.get('RESULTS','results')   # folder holding Tables/ (and Checkpoints/ for heavy.py)
CACHE=os.environ.get('CACHE','PancreasCache_112')
import json,sys,time,numpy as np,pandas as pd,torch
from pathlib import Path
torch.set_num_threads(2)
C=[''.join(c['source']) for c in json.load(open('Accuracy_Saturation_Pancreas_Hippocampus.ipynb'))['cells'] if c['cell_type']=='code']
class Cfg: pass
cfg=Cfg(); cfg.base_channels=16; cfg.n_bands=4; cfg.n_dist=3; cfg.dropout_p=0.1; cfg.device="cpu"
cfg.tta_K=6; cfg.tta_noise_sd=0.03; cfg.mc_T=10; cfg.band_dilation=3; cfg.ece_bins=10
def log(m): print(time.strftime('%H:%M:%S'),m,flush=True)
g=dict(cfg=cfg,log=log,np=np,pd=pd,torch=torch,time=time,Path=Path)
exec(C[2],g); exec(C[3],g); exec(C[7],g)
build_model=g['build_model']; prob_of=g['prob_of']; entropy_np=g['entropy_np']; fresnel_views=g['fresnel_views']
from scipy.ndimage import binary_dilation,binary_erosion,sobel
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
R=Path(OUT); CK=R/'Checkpoints'
ids=sorted(pd.read_csv(R/'Tables/pancreas_per_case_baseline_seed0.csv').case_id.unique())
LMAP={1:1,2:2}
def load(cid):
    d=np.load(f'{CACHE}/{cid}.npz'); img=d['img'].astype(np.float32); lab=d['lab'].astype(np.int64)
    return ((img-img.mean())/(img.std()+1e-6)).astype(np.float32), lab, d['img']
def net(a,s):
    n=build_model(a,cfg,3); n.load_state_dict(torch.load(CK/f'pancreas_{a}_seed{s}.pt',map_location='cpu')); n.eval(); return n
ens=[net('baseline',s) for s in range(5)]; fr=net('fresnel',0)
pool={k:[] for k in ['c1','k1','ce','ke']}; diag=[]
for i,cid in enumerate(ids):
    img,gt,raw=load(cid)
    with torch.no_grad():
        p0=prob_of(ens[0],img); pe=np.mean([prob_of(n,img) for n in ens],0)
    for p,(kc,kk) in ((p0,('c1','k1')),(pe,('ce','ke'))):
        pred=p.argmax(0); m=binary_dilation((pred>0)|(gt>0),iterations=3)
        pool[kc].append(p.max(0)[m].astype(np.float32)); pool[kk].append((pred==gt)[m])
    with torch.no_grad(): pf,coh,mi=fresnel_views(fr,img)
    pred=pf.argmax(0); m=binary_dilation((pred>0)|(gt>0),iterations=3); err=(pred!=gt)
    grad=np.sqrt(sum(sobel(raw.astype(np.float32),a)**2 for a in range(3)))
    fg=(gt>0)|(pred>0); bd=fg&~binary_erosion(fg,iterations=2); inner=binary_erosion(fg,iterations=2)
    au=roc_auc_score(err[m],coh[m]) if 0<err[m].sum()<m.sum() else np.nan
    diag.append(dict(case=cid,rho=spearmanr(coh[m],grad[m]).correlation,
        bnd=coh[bd].mean() if bd.any() else np.nan,inn=coh[inner].mean() if inner.any() else np.nan,auroc=au,
        err_mean=coh[m][err[m]].mean(),cor_mean=coh[m][~err[m]].mean()))
    log(f'{i+1}/{len(ids)} {cid}')
np.savez_compressed('pooled.npz',**{k:np.concatenate(v) for k,v in pool.items()})
pd.DataFrame(diag).to_csv('coh_diag.csv',index=False); log('DONE')
