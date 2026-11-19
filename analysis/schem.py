import os
OUT=os.environ.get('RESULTS','results')   # folder holding Tables/ (and Checkpoints/ for heavy.py)
CACHE=os.environ.get('CACHE','PancreasCache_112')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
INK="#0b0b0b"; SEC="#52514e"
def box(ax,x,y,w,h,title,lines,fc,ec,ts=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.015",fc=fc,ec=ec,lw=1.2))
    ax.text(x+0.012,y+h-0.03,title,fontsize=ts,weight="bold",color=INK,va="top")
    ax.text(x+0.012,y+h-0.085,"\n".join(lines),fontsize=8.6,color=INK,va="top",linespacing=1.45)
def arr(ax,a,b,**k): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,lw=1.2,color=SEC,**k))

# ---------- Figure 1: staged protocol ----------
fig,ax=plt.subplots(figsize=(13,5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.01,0.40,0.29,0.55,"Stage 0 — data",[
 "Pancreas: MSD Task07, 281 CT volumes",
 "  classes pancreas, mass",
 "  HU window [−100, 240], resampled to 112³",
 "Hippocampus: MSD Task04, 260 MRI volumes",
 "  classes anterior, posterior",
 "  0.5–99.5 percentile scaling, 48×64×48",
 "Fixed test set (seed 12345, 15%):",
 "  42 pancreas / 39 hippocampus cases"],"#f1f0ed","#8c8a84")
box(ax,0.355,0.40,0.29,0.55,"Stage 1 — accuracy axis",[
 "Residual 3D U-Net, swappable decoder module:",
 "  baseline · SEA · Fresnel · dropout",
 "5 seeds per arm (split + initialisation)",
 "Same optimiser, loss and stopping rule",
 "Per-case Dice averaged over seeds",
 "Wilcoxon + Holm (difference)",
 "TOST with fixed margins (equivalence)",
 "Minimum detectable difference (80% power)"],"#e3eefb","#2a78d6")
box(ax,0.70,0.40,0.29,0.55,"Stage 2 — reliability axis",[
 "Stage-1 checkpoints reused, no retraining",
 "Estimators: entropy, TTA, ensemble,",
 "  MC Dropout, Fresnel coherence,",
 "  Fresnel view disagreement",
 "Each scored on its own prediction",
 "AUROC, ECE, risk–coverage AUC",
 "Regions: boundary band, per class, volume",
 "Patient = unit; bootstrap CI; Holm"],"#e6f5ee","#1baf7a")
arr(ax,(0.302,0.675),(0.353,0.675)); arr(ax,(0.647,0.675),(0.698,0.675))
box(ax,0.355,0.03,0.29,0.30,"Question 1",[
 "Does any module improve Dice,",
 "or are the arms equivalent within",
 "a prespecified margin?"],"#ffffff","#2a78d6")
box(ax,0.70,0.03,0.29,0.30,"Question 2",[
 "Which estimator detects errors and is",
 "calibrated better than single-model",
 "entropy, and at what cost per volume?"],"#ffffff","#1baf7a")
arr(ax,(0.50,0.398),(0.50,0.335)); arr(ax,(0.845,0.398),(0.845,0.335))
box(ax,0.01,0.03,0.29,0.30,"Outputs",[
 "Per-case tables, test statistics,",
 "qualitative panels; all regenerated",
 "by one notebook from public data"],"#ffffff","#8c8a84")
fig.savefig("Framework_Overview.png",dpi=300,bbox_inches="tight"); fig.savefig("abstract_fig.png",dpi=200,bbox_inches="tight")

# ---------- Figure 2: estimator construction ----------
fig,ax=plt.subplots(figsize=(13.5,6.2)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.01,0.55,0.30,0.42,"(a) Multi-pass estimators",[
 "Entropy: H(p) of one model (1 pass)",
 "TTA: H of mean softmax over K = 6 passes",
 "  identity + 5 inputs with N(0, 0.03²) noise",
 "  and axis flips (bit pattern of k), unflipped",
 "Ensemble: H of mean softmax of M = 5",
 "  models trained with seeds 0–4 (5 passes)",
 "MC Dropout: H of mean softmax over T = 10",
 "  passes, Dropout3d p = 0.1 active in decoder"],"#e3eefb","#2a78d6")
box(ax,0.01,0.03,0.30,0.46,"(c) Scoring (every estimator)",[
 "Prediction ŷ = argmax of the estimator's",
 "  own mean probability; confidence = max p",
 "Error label e = 1[ŷ ≠ y] per voxel",
 "AUROC(u, e); ECE (10 bins);",
 "  risk–coverage AUC (20 coverage levels)",
 "Regions: band = dilate(ŷ>0 ∪ y>0, 3),",
 "  band per class, whole volume",
 "Per-case values → patient-level tests"],"#f1f0ed","#8c8a84")
# Fresnel pipeline
y0=0.80
def node(x,y,w,h,t,fc="#fff4e5",ec="#eb6834"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.012",fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8.3,color=INK)
ax.text(0.35,1.03,"(b) Single-pass Fresnel estimators (last decoder stage of the Fresnel model)",fontsize=10,weight="bold",va="top")
node(0.35,0.70,0.11,0.13,"decoder\nfeatures\nf")
node(0.50,0.70,0.10,0.13,"FFT\nF(f)")
for i,(yy,z) in enumerate([(0.84,"z₁"),(0.68,"z₂"),(0.52,"z₃")]):
    node(0.64,yy,0.15,0.11,f"× exp(i2π{z}‖k‖²)\nIFFT → view v{i+1}")
    arr(ax,(0.60,0.765),(0.64,yy+0.055))
arr(ax,(0.46,0.765),(0.50,0.765))
node(0.83,0.84,0.16,0.13,"(iv) coherence:\nVar over v₁…v₃,\nupsampled","#fdeaf1","#e87ba4")
for yy in (0.895,0.735,0.575): arr(ax,(0.79,yy),(0.83,0.905))
node(0.64,0.24,0.15,0.20,"view i alone:\nf + fuse(0,…,vᵢ,…,0)\n→ segmentation head\n→ pᵢ  (i = 1, 2, 3)")
for yy in (0.86,0.70,0.54): pass
arr(ax,(0.715,0.52),(0.715,0.44))
node(0.83,0.24,0.16,0.20,"(v) disagreement:\nH(p̄) − mean H(pᵢ)\n(mutual information)","#fdeaf1","#e87ba4")
arr(ax,(0.79,0.34),(0.83,0.34))
node(0.35,0.24,0.25,0.20,"Normal forward pass\n(all views fused) gives the\nprediction scored for both\nFresnel estimators","#f1f0ed","#8c8a84")
ax.text(0.35,0.12,"The distances z₁…z₃ are learned end-to-end; the fusion is a 1×1×1 convolution,\nso routing one view at a time isolates its contribution to the decision.",fontsize=8.2,color=SEC,va="top")
fig.savefig("Estimators_Overview.png",dpi=300,bbox_inches="tight")
