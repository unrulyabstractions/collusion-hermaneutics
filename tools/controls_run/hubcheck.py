"""hubcheck.py <hub prefix> <wild sha file>: every Hub file under the prefix has the wild source's sha256 (LFS) and the
Mac copy under data/hf has the Hub's size. No hashing of large files on the Mac."""
import sys
from huggingface_hub import HfApi
from pathlib import Path
prefix, shafile = sys.argv[1], sys.argv[2]
w = {}
for line in open(shafile):
    h, p = line.split(); w[prefix + "/" + p[2:]] = h
ok = bad = 0
for i in HfApi().list_repo_tree("unrulyabstractions/arena-capstone", repo_type="dataset", path_in_repo=prefix, recursive=True, expand=True):
    if not hasattr(i, "size"): continue
    if i.path not in w: continue
    local = Path("data/hf") / i.path
    same = (i.lfs.sha256 == w[i.path]) if i.lfs else True
    size_ok = local.exists() and local.stat().st_size == i.size
    if same and size_ok: ok += 1
    else: bad += 1; print("BAD", i.path, same, size_ok)
missing = len(w) - ok - bad
print(f"{prefix}: {ok} OK, {bad} bad, {missing} of wild's {len(w)} not on the Hub")
sys.exit(1 if bad or missing else 0)
