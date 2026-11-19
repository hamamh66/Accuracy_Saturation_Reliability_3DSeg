import os
OUT=os.environ.get('RESULTS','results')   # folder holding Tables/ (and Checkpoints/ for heavy.py)
CACHE=os.environ.get('CACHE','PancreasCache_112')
import numpy as np,pandas as pd,json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
SEC="#52514e";GRID="#e4e3df";C1="#7a7974";CE="#eb6834"
z=np.load("pooled.npz")
def rel(c,k,b=10):
    e=np.linspace(0,1,b+1); out=[]
    for i in range(b):
        m=(c>e[i])&(c<=e[i+1])
        if m.sum()>50: out.append((c[m].mean(),k[m].mean(),m.sum()))
    return np.array(out)
def rc(c,k):
    o=np.argsort(-c,kind="mergesort"); kk=k[o]; cov=np.linspace(0.05,1,40)
    return cov,np.array([1-kk[:max(1,int(len(kk)*q))].mean() for q in cov])
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
ax[0].plot([0,1],[0,1],"--",color=SEC,lw=1)
for c,k,col,lab in [(z["c1"],z["k1"],C1,"Single model"),(z["ce"],z["ke"],CE,"Ensemble (5)")]:
    r=rel(c,k); ax[0].plot(r[:,0],r[:,1],"-o",color=col,lw=2,ms=5,mec="white",label=lab)
    cov,ri=rc(c,k); ax[1].plot(cov,ri,color=col,lw=2,label=lab)
ax[0].set_xlabel("Confidence");ax[0].set_ylabel("Accuracy");ax[0].set_title("(a) Reliability diagram",loc="left",fontsize=10)
ax[1].set_xlabel("Coverage (most confident fraction kept)");ax[1].set_ylabel("Error rate");ax[1].set_title("(b) Risk–coverage",loc="left",fontsize=10)
for a in ax: a.grid(color=GRID,lw=0.8); a.set_axisbelow(True); a.legend(frameon=False)
ax[0].set_xlim(0.3,1.0);ax[0].set_ylim(0.3,1.0)
fig.tight_layout(); fig.savefig("results/Figures/reliability_riskcov.png",dpi=300,bbox_inches="tight")
cs,r1=rc(z["c1"],z["k1"]); _,re_=rc(z["ce"],z["ke"])
better=(re_<r1).mean()
print("rc pooled better fraction",better, r1[[0,9,19,39]], re_[[0,9,19,39]])
# over-confidence check
r=rel(z["c1"],z["k1"]); print("single conf-acc",np.round(r[:,0]-r[:,1],3)); r2=rel(z["ce"],z["ke"]); print("ens",np.round(r2[:,0]-r2[:,1],3))
D=pd.read_csv("coh_diag.csv"); print(D.describe().round(3))
pos=(D.rho>0).mean(); bnd=(D.bnd>D.inn).mean(); errhi=(D.err_mean>D.cor_mean).mean()
print("rho>0",pos,"bnd>inn",bnd,"err>cor",errhi,"mean rho",D.rho.mean(),"auroc",D.auroc.mean())
fig,ax=plt.subplots(1,3,figsize=(11,3.2))
ax[0].hist(D.rho,bins=12,color="#2a78d6",edgecolor="white"); ax[0].axvline(0,color=SEC,lw=1)
ax[0].set_xlabel("Spearman ρ (coherence vs gradient)");ax[0].set_ylabel("Cases");ax[0].set_title("(a) Tracks contrast",loc="left",fontsize=10)
lim=[min(D.bnd.min(),D.inn.min()),max(D.bnd.max(),D.inn.max())]
ax[1].plot(lim,lim,"--",color=SEC,lw=1); ax[1].scatter(D.inn,D.bnd,s=22,color="#eb6834",edgecolor="white")
ax[1].set_xlabel("Mean coherence, interior");ax[1].set_ylabel("Mean coherence, boundary");ax[1].set_title("(b) Boundary vs interior",loc="left",fontsize=10)
ax[2].hist(D.auroc,bins=12,color="#eda100",edgecolor="white"); ax[2].axvline(0.5,color=SEC,lw=1,ls="--")
ax[2].set_xlabel("Failure-detection AUROC");ax[2].set_ylabel("Cases");ax[2].set_title("(c) Does not detect errors",loc="left",fontsize=10)
for a in ax: a.grid(color=GRID,lw=0.8); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig("results/Figures/coherence_diagnostic_panel.png",dpi=300,bbox_inches="tight")
json.dump(dict(pos=pos,bnd=bnd,errhi=errhi,rho=D.rho.mean(),au=D.auroc.mean(),better=better),open("diag.json","w"))
