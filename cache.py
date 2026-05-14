"""
cache.py — Pickle-based disk cache keyed on (task_id, rules).

Avoids re-running the model+tests for the same (problem, rules) pair
across runs and across the same run.
"""

import hashlib
import pickle

from config import CACHE_DIR

BASELINE_KEY = "__baseline__"


def _cache_key(task_id: str, rules: str) -> str:
    return hashlib.md5(f"{task_id}||{rules}".encode()).hexdigest()


def cache_get(task_id: str, rules: str):
    path = CACHE_DIR / (_cache_key(task_id, rules) + ".pkl")
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass
    return None


def cache_set(task_id: str, rules: str, value) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / (_cache_key(task_id, rules) + ".pkl")
    path.write_bytes(pickle.dumps(value))
