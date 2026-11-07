import os, fnmatch, collections, sys
from huggingface_hub import HfApi
REPO="unrulyabstractions/arena-capstone"; data="data/hf"
api=HfApi()
info=api.dataset_info(REPO, files_metadata=True)
print("HF head commit:", info.sha, info.last_modified)
files=[(s.rfilename, s.size) for s in info.siblings]
print("total files on HF:", len(files))
# top-level groups
grp=collections.Counter(); miss=collections.Counter(); diff=collections.Counter(); missbytes=collections.Counter()
examples=collections.defaultdict(list)
for f,sz in files:
    parts=f.split("/")
    g="/".join(parts[:3]) if parts[0] in ("activations_extra","activations") else "/".join(parts[:4]) if parts[0]=="spar-guardian-agents" else "/".join(parts[:2])
    grp[g]+=1
    p=os.path.join(data,f)
    if not os.path.exists(p):
        miss[g]+=1; missbytes[g]+=sz or 0
        if len(examples[g])<3: examples[g].append(f)
    elif sz is not None and os.path.getsize(p)!=sz:
        diff[g]+=1
        if len(examples[g])<3: examples[g].append(f+" (size differs)")
for g in sorted(grp):
    if miss[g] or diff[g]:
        print(f"{g:70s} HF {grp[g]:6d}  missing {miss[g]:6d} ({missbytes[g]/1e9:.2f} GB)  size-diff {diff[g]}")
        for e in examples[g]: print("     e.g.", e)
print("--- groups fully present locally:")
for g in sorted(grp):
    if not miss[g] and not diff[g]: print("   ", g, grp[g])
