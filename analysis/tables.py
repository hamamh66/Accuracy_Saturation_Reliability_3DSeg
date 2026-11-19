import os
OUT=os.environ.get('RESULTS','results')   # folder holding Tables/ (and Checkpoints/ for heavy.py)
CACHE=os.environ.get('CACHE','PancreasCache_112')
import pandas as pd,numpy as np,re
T=OUT+"/Tables/"
S=pd.read_csv("s1_summary.csv");R=pd.read_csv("s1_tests.csv")
N={"baseline":"Baseline","sea":"SEA","fresnel":"Fresnel","dropout":"Dropout"}
def f(x,d=3): return f"{x:.{d}f}".replace("-","$-$")
def pv(p): return "$<0.001$" if p<0.001 else f"{p:.3f}"
L=[]
L.append(r"\begin{tabular}{llcccccc}\toprule")
L.append(r"Task / class & Arm & Dice (mean\,$\pm$\,seed SD) & $\Delta$ [90\% CI] & $p_{\mathrm{Holm}}$ & seeds$+$ & $p_{\mathrm{TOST,Holm}}$ (margin) & MDD$_{80}$\\\midrule")
for task,cls in [("pancreas",["pancreas","mass"]),("hippocampus",["anterior","posterior"])]:
    for k in cls:
        for a in ["baseline","sea","fresnel","dropout"]:
            s=S[(S.task==task)&(S.klass==k)&(S.arch==a)].iloc[0]
            lab=f"{task.capitalize()} / {k}" if a=="baseline" else ""
            if a=="baseline": L.append(f"{lab} & {N[a]} & ${s['mean']:.3f}\\pm{s.seed_sd:.3f}$ & -- & -- & -- & -- & --\\\\")
            else:
                r=R[(R.task==task)&(R.klass==k)&(R.arch==a)].iloc[0]
                L.append(f" & {N[a]} & ${s['mean']:.3f}\\pm{s.seed_sd:.3f}$ & {f(r.delta)} [{f(r.lo)}, {f(r.hi)}] & {pv(r.p_holm)} & {int(r.seeds_pos)}/5 & {pv(r.p_tost_holm)} ($\\pm{r.margin:.2f}$) & {r.mdd:.3f}\\\\")
        L.append(r"\addlinespace")
L[-1]=r"\bottomrule\end{tabular}"
open("tab_stage1.tex","w").write("\n".join(L))
EN={"entropy":"Entropy","tta":"TTA","ensemble":"Deep ensemble","mc_dropout":"MC Dropout","fresnel_disagree":"Fresnel disagreement","fresnel_coh":"Fresnel coherence"}
PASS={"entropy":"1","tta":"6","ensemble":"5","mc_dropout":"10","fresnel_disagree":"1","fresnel_coh":"1"}
L=[r"\begin{tabular}{lllccccc}\toprule",r"Task & Model & Estimator & AUROC [95\% CI] $\uparrow$ & ECE $\downarrow$ & RC-AUC $\downarrow$ & Passes & s/volume\\\midrule"]
for task in ["pancreas","hippocampus"]:
    s=pd.read_csv(T+f"{task}_stage2_summary.csv");s=s[s.region=="band"]
    t=pd.read_csv(T+f"{task}_stage2_tests.csv");t=t[t.region=="band"]
    for a in ["baseline","fresnel"]:
        for e in ["entropy","tta","ensemble","mc_dropout"]+(["fresnel_disagree","fresnel_coh"] if a=="fresnel" else []):
            r=s[(s.assess==a)&(s.method==e)].iloc[0]
            def mark(m):
                q=t[(t.assess==a)&(t.method==e)&(t.metric==m)]
                if not len(q): return ""
                q=q.iloc[0]
                if q.p_holm>=0.05: return ""
                good=(q.delta>0) if m=="auroc" else (q.delta<0)
                return "$^{\\ast}$" if good else "$^{\\dagger}$"
            ece=f"{r.ece:.3f}{mark('ece')}" if not e.startswith("fresnel") else "(= entropy)"
            rc=f"{r.rcauc:.3f}{mark('rcauc')}" if not e.startswith("fresnel") else "(= entropy)"
            L.append(f"{task.capitalize() if (a=='baseline' and e=='entropy') else ''} & {N[a] if e=='entropy' else ''} & {EN[e]} & {r.auroc:.3f} [{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]{mark('auroc')} & {ece} & {rc} & {PASS[e]} & {r.seconds_per_volume:.3f}\\\\")
        L.append(r"\addlinespace")
L[-1]=r"\bottomrule\end{tabular}"
open("tab_stage2.tex","w").write("\n".join(L))
# per-class table baseline model
L=[r"\begin{tabular}{llcccc}\toprule",r"Task & Estimator & AUROC class 1 & AUROC class 2 & ECE class 1 & ECE class 2\\\midrule"]
for task,cn in [("pancreas",("pancreas","mass")),("hippocampus",("anterior","posterior"))]:
    s=pd.read_csv(T+f"{task}_stage2_summary.csv");t=pd.read_csv(T+f"{task}_stage2_tests.csv")
    for i,e in enumerate(["entropy","tta","ensemble","mc_dropout"]):
        cells=[]
        for m in ["auroc","ece"]:
            for reg in ["band_c1","band_c2"]:
                r=s[(s.assess=="baseline")&(s.method==e)&(s.region==reg)].iloc[0]
                q=t[(t.assess=="baseline")&(t.method==e)&(t.metric==m)&(t.region==reg)]
                mk=""
                if len(q) and q.iloc[0].p_holm<0.05: mk="$^{\\ast}$" if ((q.iloc[0].delta>0)==(m=="auroc")) else "$^{\\dagger}$"
                cells.append(f"{r[m]:.3f}{mk}")
        L.append(f"{(task.capitalize()+' ('+cn[0]+' / '+cn[1]+')') if i==0 else ''} & {EN[e]} & "+" & ".join(cells)+"\\\\")
    L.append(r"\addlinespace")
L[-1]=r"\bottomrule\end{tabular}"
open("tab_perclass.tex","w").write("\n".join(L))
print(open("tab_stage1.tex").read());print(open("tab_stage2.tex").read());print(open("tab_perclass.tex").read())
# epochs
log=open(OUT+"/run_log.txt").read() if os.path.exists(OUT+"/run_log.txt") else ""
st=[int(x) for x in re.findall(r"early stop at epoch (\d+)",log)]
if st: print("stops",len(st),min(st),max(st),np.median(st))
