"""
config.py — All tunable settings in one place.

Change the model by editing MODEL_NAME.
Change the benchmark size by editing NUM_PROBLEMS / VAL_SIZE.
Change the LCB release by editing LCB_VERSION_TAG.
"""

from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 30

# ── Generation ────────────────────────────────────────────────────────────────

GEN_TEMPERATURE = 0.1

# ── GEPA loop ─────────────────────────────────────────────────────────────────

NUM_PROBLEMS     = 30
GENERATIONS      = 3
MINIBATCH_SIZE   = 5  # nb de problèmes montrés au mutator — réduit pour que le prompt tienne dans le contexte 32K du modèle
VAL_SIZE         = 10
EVAL_TIMEOUT     = 100
POPULATION_SIZE  = 1
PARETO_SIZE      = 6
MUTATION_RATE    = 0.7
PERFECT_SCORE    = 1.0
MAX_MERGE_INVOCATIONS    = 5
MERGE_VAL_OVERLAP_FLOOR  = 1
USE_MERGE        = False  # whether to use the LLM-based merge step in GEPA (ablation)


# ── Benchmark ─────────────────────────────────────────────────────────────────

LCB_DATASET  = "livecodebench/code_generation_lite"
LCB_VERSION_TAG    = "release_v2"
LCB_FALLBACK_REPO  = "bzantium/livecodebench"

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results_gepa_lcb_v13")
CACHE_DIR = Path(".gepa_cache_lcb_v13")
