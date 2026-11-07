"""Hermeneutic cheating probe: other models' transcripts read through Qwen3.8-27B's residual stream."""
import os

# The controls data: <DATA_ROOT>/<host>/runs/<run_name>/ and <DATA_ROOT>/<host>/logs/<run_name>/*.eval. Read-only.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("HERM_DATA", os.path.join(REPO, "data/apart_controls"))
# Qwen3.8-27B tokenizer (and, on the GPU host, the weights): a local snapshot directory
MODEL_DIR = os.environ.get("HERM_MODEL", os.path.join(REPO, "data/tokenizer/Qwen3.8-27B"))
