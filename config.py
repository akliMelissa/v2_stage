"""
config.py — All tunable settings in one place.

Change the model by editing MODEL_NAME.
Change the benchmark size by editing NUM_PROBLEMS / VAL_SIZE.
Change the LCB release by editing LCB_VERSION_TAG.
"""

import os
from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 24

# ── Generation ────────────────────────────────────────────────────────────────

GEN_TEMPERATURE = 0.1

# ── GEPA loop ─────────────────────────────────────────────────────────────────

NUM_PROBLEMS   = 511
GENERATIONS    = 15
MINIBATCH_SIZE = 30
VAL_SIZE       = 100
EVAL_TIMEOUT   = 100


# ── Benchmark ─────────────────────────────────────────────────────────────────

LCB_DATASET  = "livecodebench/code_generation_lite"
LCB_VERSION_TAG    = "release_v2"
LCB_FALLBACK_REPO  = "bzantium/livecodebench"

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results_gepa_lcb_v12")
CACHE_DIR = Path(".gepa_cache_lcb_v12")
