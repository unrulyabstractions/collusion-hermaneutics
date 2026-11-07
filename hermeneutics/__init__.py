"""Hermeneutic cheating probe: other models' transcripts read through Qwen3.8-27B's residual stream."""
import os

# BLAS and OpenMP thread caps before numpy or torch load anywhere in the package: forked workers otherwise spin one
# thread pool per core each (a 256-core host reached a load average above 1,000 on 2026-11-07).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

# The controls data: <DATA_ROOT>/<host>/runs/<run_name>/ and <DATA_ROOT>/<host>/logs/<run_name>/*.eval. Read-only.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("HERM_DATA", os.path.join(REPO, "data/apart_controls"))
# Qwen3.8-27B tokenizer (and, on the GPU host, the weights): a local snapshot directory
MODEL_DIR = os.environ.get("HERM_MODEL", os.path.join(REPO, "data/tokenizer/Qwen3.8-27B"))
