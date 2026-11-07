import numpy as np, pandas as pd, time, sys
sys.argv=["x"]
from hermeneutics import probes as P
rng=np.random.default_rng(0)
authors=["ds","gpt5m","haiku","luna","qwen"]; tasks=list("abcdefghi")
rows=[]; acts={26:[],38:[],51:[]}
d=5120
for i in range(150):
    au=authors[i%5]; cond=["solo","cheat"][(i//5)%2]; y=int(rng.random()<0.5)
    nt=int(rng.integers(3,12))
    rows.append({"run":f"r{i}","author":au,"condition":cond,"task":tasks[i%9],"y":y,"rung":"x","n_turns":nt,"n_readable":nt,"n_tokens":1000})
    for L in acts:
        a=rng.normal(size=(nt,2,d)).astype(np.float32)
        a[:,:,0]+=5*authors.index(au); a[:,:,1]+=1.5*y; a[:,:,2]+=1.0*(cond=="cheat")
        acts[L].append(a)
agents=pd.DataFrame(rows)
t0=time.time()
X,turn_of=P.features(agents,acts,38,"close","agent"); print("agent X",X.shape, flush=True)
res=P.analyze(X,agents,None,n_shuf=4,jobs=1,full=True)
print("author probe", {k:round(v["accuracy"],3) for k,v in res["author_probe"].items()})
print("cheat rows", len(res["cheat"]), "shuffle keys", list(res["shuffle"]), "nuisance", {k:round(v,3) for k,v in res["nuisance"].items()})
print([(r["features"], r["scope"], round(r["auroc"],3)) for r in res["cheat"] if r["scope"] in ("pooled","LOTO ds","ds")])
print("erasure", res["erasure_check"])
Xt,turn_of=P.features(agents,acts,"all","both","turn"); print("turn X",Xt.shape, flush=True)
res2=P.analyze(Xt,agents,turn_of,n_shuf=0,jobs=1,full=False)
print([(r["features"], r["scope"], round(r["auroc"],3), round(r.get("auroc_turns",0),3)) for r in res2["cheat"][:3]])
print("flat rows", len(P.flat_rows(res,"solo",38,"close","agent")), f"{time.time()-t0:.0f}s")
