"""
config.py — All tunable settings in one place.

Change the model by editing MODEL_NAME.
Change the benchmark size by editing NUM_PROBLEMS / VAL_SIZE.
Change the LCB release by editing LCB_VERSION_TAG.
"""

import os
from pathlib import Path

import torch

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
CUDA_DEVICE = int(os.environ.get("CUDA_DEVICE", "0"))
DEVICE = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"

# ── Generation ────────────────────────────────────────────────────────────────

MAX_NEW_TOKENS = 1500
GEN_TEMPERATURE = 0.1

# ── GEPA loop ─────────────────────────────────────────────────────────────────

NUM_PROBLEMS   = 30
GENERATIONS    = 5
MINIBATCH_SIZE = 10
VAL_SIZE       = 15
EVAL_TIMEOUT   = 30

# ── Benchmark ─────────────────────────────────────────────────────────────────

LCB_DATASET        = "livecodebench/code_generation_lite"
LCB_VERSION_TAG    = "release_v2"
LCB_FALLBACK_REPO  = "bzantium/livecodebench"

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results_gepa_lcb_v3")
CACHE_DIR   = Path(".gepa_cache_lcb_v3")
