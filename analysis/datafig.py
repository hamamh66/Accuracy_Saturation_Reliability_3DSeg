import os
OUT=os.environ.get('RESULTS','results')   # folder holding Tables/ (and Checkpoints/ for heavy.py)
CACHE=os.environ.get('CACHE','PancreasCache_112')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
INK="#0b0b0b";SEC="#52514e";GRID="#e4e3df"
COL={"fresnel":"#eb6834","sea":"#2a78d6","dropout":"#1baf7a"}
NAME={"fresnel":"Fresnel","sea":"SEA","dropout":"Dropout"}
R=pd.read_csv("s1_tests.csv")
fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharey=False)
for ax,(task,cls) in zip(axs,[("pancreas",["pancreas","mass"]),("hippocampus",["anterior","posterior"])]):
    y=0;ticks=[];lab=[]
    for k in cls:
        m=R[(R.task==task)&(R.klass==k)].margin.iloc[0]
        for a in ["sea","fresnel","dropout"]:
            r=R[(R.task==task)&(R.klass==k)&(R.arch==a)].iloc[0]
            ax.plot([r.lo,r.hi],[y,y],color=COL[a],lw=2,solid_capstyle="round")
            ax.plot(r.delta,y,"o",ms=7,color=COL[a],mec="white",mew=1.5)
            ax.plot([-m,-m],[y-0.4,y+0.4],color=SEC,lw=1,ls=":");ax.plot([m,m],[y-0.4,y+0.4],color=SEC,lw=1,ls=":")
            ticks.append(y);lab.append(f"{k} · {NAME[a]}");y+=1
        y+=0.6
    ax.axvline(0,color=SEC,lw=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_yticks(ticks);ax.set_yticklabels(lab);ax.invert_yaxis()
    ax.set_xlabel("Δ Dice vs baseline (mean, 90% CI)",color=SEC)
    ax.set_title(f"{task.capitalize()} (n = {int(R[R.task==task].n.iloc[0])} test cases, 5 seeds)",fontsize=10,loc="left",color=INK)
    ax.grid(axis="x",color=GRID,lw=0.8);ax.set_axisbelow(True)
axs[0].text(0.99,0.02,"dotted: equivalence margin",transform=axs[0].transAxes,ha="right",fontsize=8,color=SEC)
fig.tight_layout();fig.savefig("stage1_equivalence.png",dpi=300,bbox_inches="tight")

# Stage-2: AUROC and ECE with bootstrap CI, baseline-assessed, band region
E=["entropy","tta","ensemble","mc_dropout","fresnel_disagree","fresnel_coh"]
EN={"entropy":"Entropy (1 pass)","tta":"TTA (6)","ensemble":"Ensemble (5)","mc_dropout":"MC Dropout (10)","fresnel_disagree":"Fresnel disagreement (1)","fresnel_coh":"Fresnel coherence (1)"}
EC={"entropy":"#7a7974","tta":"#2a78d6","ensemble":"#eb6834","mc_dropout":"#1baf7a","fresnel_disagree":"#e87ba4","fresnel_coh":"#eda100"}
T=OUT+"/Tables/"
fig,axs=plt.subplots(2,2,figsize=(10,5.6),gridspec_kw=dict(wspace=0.06,hspace=0.45)); axs=axs.ravel()
for j,task in enumerate(["pancreas","hippocampus"]):
    s=pd.read_csv(T+f"{task}_stage2_summary.csv");s=s[s.region=="band"]
    for i,(met,lab) in enumerate([("auroc","Failure-detection AUROC ↑"),("ece","ECE ↓")]):
        ax=axs[2*j+i]
        for y,e in enumerate(E):
            a="fresnel" if e.startswith("fresnel") else "baseline"
            r=s[(s.assess==a)&(s.method==e)]
            if met=="ece" and e.startswith("fresnel"): continue
            r=r.iloc[0]
            ax.plot([r[met+"_ci_low"],r[met+"_ci_high"]],[y,y],color=EC[e],lw=2.2,solid_capstyle="round")
            ax.plot(r[met],y,"o",ms=7,color=EC[e],mec="white",mew=1.5)
        ref=s[(s.assess=="baseline")&(s.method=="entropy")].iloc[0][met]
        ax.axvline(ref,color=SEC,lw=0.8,ls=":")
        ax.set_yticks(range(len(E)));ax.set_yticklabels([EN[e] for e in E] if i==0 else [])
        ax.invert_yaxis();ax.grid(axis="x",color=GRID,lw=0.8);ax.set_axisbelow(True)
        ax.set_title(f"{task.capitalize()}: {lab}",fontsize=9.5,loc="left")
        if met=="auroc": ax.axvline(0.5,color=GRID,lw=1)
fig.savefig("stage2_estimators.png",dpi=300,bbox_inches="tight")
