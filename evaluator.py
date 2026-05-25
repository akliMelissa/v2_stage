"""
evaluator.py — Turn a (problem, rules) pair into a pass/fail result.

NEW: Includes answer leakage detection to flag mutations that embed
code into prompts.

Steps:
  1. Build the full original prompt.
  2. Apply transformation rules via LLM.
  3. CHECK FOR ANSWER LEAKAGE (new).
  4. Ask the LLM for code from the improved prompt.
  5. Run the tests via eval_benchmark.py-style harness.

Result fields match eval_benchmark.py:
  Pass@1, Tests_Passed, n_Tests, Eval_Status
plus GEPA bookkeeping (success/error/prompt/improved/code/task_id).
"""

import re
from concurrent.futures import ThreadPoolExecutor

from config import GEN_TEMPERATURE
from cache import cache_get, cache_set, BASELINE_KEY
from code_extraction import extract_code
from model import safe_call, batch_call_llm, CODE_GEN_SYSTEM_PROMPT
from prompts import APPLY_RULES_PROMPT
from test_runner import run_tests


# ── Answer leakage detection ───────────────────────────────────────────────


def _check_answer_leakage(original: str, improved: str) -> bool:
    """Flag if transformation added actual code blocks or function definitions."""
    import re # Au cas où re n'est pas importé dans ce fichier
    
    # On cherche uniquement des structures de code Python réelles sur l'ensemble du texte
    leak_patterns = [
        r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?\s*:\s*\n\s+',  # Une fonction avec son bloc indenté
        r'class\s+Solution\s*:',                                  # Une classe de résolution
        r'```python\n\s*def\s+',                                  # Un bloc Markdown python qui démarre une fonction
    ]
    
    for pattern in leak_patterns:
        if re.search(pattern, improved):  # Analyse globale sur 'improved' (sans découpage)
            return True
            
    return False

# ── Prompt transformation ──────────────────────────────────────────────────

def apply_transformation_rules(rules: str, original_prompt: str) -> str:
    """Run the rules over the original prompt and return the improved version."""
    prompt = APPLY_RULES_PROMPT.format(rules=rules, original_prompt=original_prompt)
    improved = safe_call(prompt, temperature=0.3)
    return improved


# ── Code generation ────────────────────────────────────────────────────────

_CODE_GEN_SUFFIX = (
    "\n\nWrite a complete, fully implemented Python solution. "
    "Do NOT use `pass`, empty function bodies, or placeholder docstrings — every function must contain real logic. "
    "Always include ALL necessary imports at the top (e.g. from functools import lru_cache, "
    "from collections import defaultdict, import heapq, etc.). "
    "Wrap your solution in a ```python ... ``` code block."
)


def generate_code(prompt: str, starter_code: str = "",
                  temperature: float = GEN_TEMPERATURE) -> str:
    """Generate code from a prompt, attaching starter code when present."""
    if starter_code and starter_code.strip():
        full_prompt = (
            f"{prompt}\n\n"
            f"Complete this starter code:\n```python\n{starter_code}\n```"
            f"{_CODE_GEN_SUFFIX}"
        )
    else:
        full_prompt = prompt + _CODE_GEN_SUFFIX

    return extract_code(
        safe_call(full_prompt, temperature=temperature,
                  system_prompt=CODE_GEN_SYSTEM_PROMPT)
    )


# ── Per-problem evaluation ─────────────────────────────────────────────────

def _build_full_prompt(problem: dict) -> str:
    """The prompt as the model would see it before any transformation."""
    return problem.get("question_content", "") or problem.get("original_prompt", "")


def _make_result(p: dict, original: str, improved: str, code: str,
                 pass_at_1: bool, status: str, tests_passed: int, n_tests: int) -> dict:
    """Shape one evaluation outcome into a dict.
    
    Fields named like eval_benchmark.py (Pass@1, Tests_Passed, n_Tests, Eval_Status),
    with `success` and `error` for GEPA loop compatibility.
    """
    return {
        "success":            pass_at_1,
        "error":              "" if pass_at_1 else status,
        "Pass@1":             pass_at_1,
        "Tests_Passed":       tests_passed,
        "n_Tests":            n_tests,
        "Eval_Status":        status,
        "prompt":             original,
        "improved":           improved,
        "code":               code,
        "task_id":            p["task_id"],
        "canonical_solution": p.get("canonical_solution", ""),
    }


def evaluate_one(p: dict, rules: str) -> dict:
    """Evaluate one problem under one rule set. Cached. Returns a result dict.
    
    NEW: Includes answer leakage check after transformation.
    """
    cached = cache_get(p["task_id"], rules)
    if cached is not None and isinstance(cached, dict) and "success" in cached:
        # Backfill canonical_solution for older cached entries
        if "canonical_solution" not in cached:
            cached["canonical_solution"] = p.get("canonical_solution", "")
        return cached

    original = _build_full_prompt(p)
    improved_prompt = apply_transformation_rules(rules, original)

    # ← NEW: Check for answer leakage (code embedded in prompt)
    if _check_answer_leakage(original, improved_prompt):
        result = _make_result(p, original, improved_prompt, "", False,
                             "Answer leakage detected in transformation", 0, 0)
        cache_set(p["task_id"], rules, result)
        return result

    starter = p.get("starter_code", "") or ""
    code = generate_code(improved_prompt, starter_code=starter)
    pass_at_1, status, tests_passed, n_tests = run_tests(code, p)

    result = _make_result(p, original, improved_prompt, code,
                          pass_at_1, status, tests_passed, n_tests)
    cache_set(p["task_id"], rules, result)
    return result


def evaluate_baseline(p: dict) -> dict:
    """Evaluate one problem with the original prompt — no transformation. Cached."""
    cached = cache_get(p["task_id"], BASELINE_KEY)
    if cached is not None and isinstance(cached, dict) and "success" in cached:
        if "canonical_solution" not in cached:
            cached["canonical_solution"] = p.get("canonical_solution", "")
        return cached

    original = _build_full_prompt(p)
    starter = p.get("starter_code", "") or ""
    code = generate_code(original, starter_code=starter, temperature=GEN_TEMPERATURE)
    pass_at_1, status, tests_passed, n_tests = run_tests(code, p)

    result = _make_result(p, original, original, code,
                          pass_at_1, status, tests_passed, n_tests)
    cache_set(p["task_id"], BASELINE_KEY, result)
    return result


def evaluate_batch(problems: list[dict], rules: str) -> list[dict]:
    """Evaluate a batch: 2 batched GPU calls + parallel test execution."""
    results = [None] * len(problems)
    uncached_idx = []

    for i, p in enumerate(problems):
        cached = cache_get(p["task_id"], rules)
        if cached is not None and isinstance(cached, dict) and "success" in cached:
            if "canonical_solution" not in cached:
                cached["canonical_solution"] = p.get("canonical_solution", "")
            results[i] = cached
        else:
            uncached_idx.append(i)

    if not uncached_idx:
        return results

    probs = [problems[i] for i in uncached_idx]
    originals = [_build_full_prompt(p) for p in probs]

    # Phase 1: batch transform (1 GPU pass)
    t_prompts = [APPLY_RULES_PROMPT.format(rules=rules, original_prompt=o) for o in originals]
    improved_raw = batch_call_llm(t_prompts, temperature=0.3)
    improved = []
    for orig, imp in zip(originals, improved_raw):
        # On nettoie juste les espaces blancs invisibles au début/fin
        imp_clean = imp.strip() if imp else ""
        # On garde le prompt tel quel s'il fait au moins 20 caractères, sinon fallback
        improved.append(imp_clean if len(imp_clean) >= 20 else orig)


    # Phase 2: batch code gen (1 GPU pass, skip leaked)
    leaked = [_check_answer_leakage(o, i) for o, i in zip(originals, improved)]
    c_prompt_idx, c_prompts = [], []
    for j, (p, imp, lk) in enumerate(zip(probs, improved, leaked)):
        if not lk:
            starter = p.get("starter_code", "") or ""
            fp = (f"{imp}\n\nComplete this starter code:\n```python\n{starter}\n```{_CODE_GEN_SUFFIX}"
                  if starter.strip() else imp + _CODE_GEN_SUFFIX)
            c_prompt_idx.append(j)
            c_prompts.append(fp)

    raw_codes = (batch_call_llm(c_prompts, temperature=GEN_TEMPERATURE,
                                system_prompt=CODE_GEN_SYSTEM_PROMPT)
                 if c_prompts else [])
    codes = [""] * len(probs)
    for k, j in enumerate(c_prompt_idx):
        codes[j] = extract_code(raw_codes[k])

    # Phase 3: parallel test execution (CPU)
    def _run(j):
        p = probs[j]
        orig, imp, code, lk = originals[j], improved[j], codes[j], leaked[j]
        if lk:
            r = _make_result(p, orig, imp, "", False,
                             "Answer leakage detected in transformation", 0, 0)
        else:
            pass_at_1, status, tp, nt = run_tests(code, p)
            r = _make_result(p, orig, imp, code, pass_at_1, status, tp, nt)
        cache_set(p["task_id"], rules, r)
        return j, r

    with ThreadPoolExecutor(max_workers=min(len(probs), 16)) as ex:
        for j, r in ex.map(_run, range(len(probs))):
            results[uncached_idx[j]] = r

    return results


def evaluate_baseline_batch(problems: list[dict]) -> list[dict]:
    """Baseline evaluation: 1 batched GPU call + parallel test execution."""
    results = [None] * len(problems)
    uncached_idx = []

    for i, p in enumerate(problems):
        cached = cache_get(p["task_id"], BASELINE_KEY)
        if cached is not None and isinstance(cached, dict) and "success" in cached:
            if "canonical_solution" not in cached:
                cached["canonical_solution"] = p.get("canonical_solution", "")
            results[i] = cached
        else:
            uncached_idx.append(i)

    if not uncached_idx:
        print("  [baseline] all cached, skipping GPU calls.")
        return results

    probs = [problems[i] for i in uncached_idx]
    originals = [_build_full_prompt(p) for p in probs]

    c_prompts = []
    for p, orig in zip(probs, originals):
        starter = p.get("starter_code", "") or ""
        fp = (f"{orig}\n\nComplete this starter code:\n```python\n{starter}\n```{_CODE_GEN_SUFFIX}"
              if starter.strip() else orig + _CODE_GEN_SUFFIX)
        c_prompts.append(fp)

    raw_codes = batch_call_llm(c_prompts, temperature=GEN_TEMPERATURE,
                               system_prompt=CODE_GEN_SYSTEM_PROMPT)
    codes = [extract_code(r) for r in raw_codes]

    def _run(j):
        p = probs[j]
        orig, code = originals[j], codes[j]
        pass_at_1, status, tp, nt = run_tests(code, p)
        r = _make_result(p, orig, orig, code, pass_at_1, status, tp, nt)
        cache_set(p["task_id"], BASELINE_KEY, r)
        return j, r

    with ThreadPoolExecutor(max_workers=min(len(probs), 16)) as ex:
        for j, r in ex.map(_run, range(len(probs))):
            results[uncached_idx[j]] = r

    return results


def score_transformation_rules(rules: str, problems: list) -> tuple[int, list[dict]]:
    """Run a rule set over a list of problems. Returns (pass_count, results)."""
    results = evaluate_batch(problems, rules)
    success_count = sum(1 for r in results if r["success"])
    return success_count, results