import os
OUT=os.environ.get('RESULTS','results')   # folder holding Tables/ (and Checkpoints/ for heavy.py)
CACHE=os.environ.get('CACHE','PancreasCache_112')
import pandas as pd,numpy as np,glob,re
from scipy import stats
T=OUT+"/Tables/"
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*p[i])); adj[i]=run
    return adj
def tost(d,margin):
    n=len(d); m=d.mean(); se=d.std(ddof=1)/np.sqrt(n)
    return max(1-stats.t.cdf((m+margin)/se,n-1),stats.t.cdf((m-margin)/se,n-1)), stats.t.interval(0.90,n-1,loc=m,scale=se)
def mdd(d): return (stats.norm.ppf(0.975)+stats.norm.ppf(0.8))*d.std(ddof=1)/np.sqrt(len(d))
M={"pancreas":0.03,"mass":0.05,"anterior":0.02,"posterior":0.02}
out=[];summ=[]
for task,arches in [("pancreas",["fresnel","sea","dropout"]),("hippocampus",["fresnel","sea","dropout"])]:
    df=pd.concat([pd.read_csv(f) for f in glob.glob(T+f"{task}_per_case_*_seed*.csv")])
    df=df[df.arch!="edgeatt"]
    for (a,k),g in df.groupby(["arch","klass_name"]):
        pc=g.groupby("case_id").dice.mean(); sd=g.groupby("seed").dice.mean()
        summ.append(dict(task=task,arch=a,klass=k,n=pc.size,mean=pc.mean(),seed_sd=sd.std(),hd95=g.groupby("case_id").hd95.mean().mean()))
    rows=[]
    for k in sorted(df.klass_name.unique()):
        b=df[(df.arch=="baseline")&(df.klass_name==k)].groupby("case_id").dice.mean()
        for a in arches:
            r=df[(df.arch==a)&(df.klass_name==k)].groupby("case_id").dice.mean(); c=b.index.intersection(r.index); d=(r[c]-b[c]).values
            p=stats.wilcoxon(r[c],b[c]).pvalue; pt,ci=tost(d,M[k])
            seeds=sum(1 for s in range(5) if (df[(df.arch==a)&(df.seed==s)&(df.klass_name==k)].set_index("case_id").dice-df[(df.arch=="baseline")&(df.seed==s)&(df.klass_name==k)].set_index("case_id").dice).mean()>0)
            rows.append(dict(task=task,klass=k,arch=a,n=len(c),delta=d.mean(),lo=ci[0],hi=ci[1],p=p,margin=M[k],p_tost=pt,mdd=mdd(d),seeds_pos=seeds))
    r=pd.DataFrame(rows); r["p_holm"]=holm(r.p); r["p_tost_holm"]=holm(r.p_tost); r["equiv"]=r.p_tost_holm<0.05; out.append(r)
S=pd.DataFrame(summ); R=pd.concat(out)
pd.set_option("display.width",250)
print(S.round(4).to_string(index=False)); print(R.round(4).to_string(index=False))
S.to_csv("s1_summary.csv",index=False); R.to_csv("s1_tests.csv",index=False)
