"""
evaluator.py — Turn a (problem, rules) pair into a pass/fail result.

Steps:
  1. Build the full original prompt.
  2. Apply transformation rules via LLM.
  3. Check for answer leakage (code embedded in the improved prompt).
  4. Ask the LLM for code from the improved prompt.
  5. Run the tests and return pass/fail.
"""

import re
from concurrent.futures import ThreadPoolExecutor

from config import GEN_TEMPERATURE
from cache import cache_get, cache_set, BASELINE_KEY
from code_extraction import extract_code
from model import safe_call, batch_call_llm, CODE_GEN_SYSTEM_PROMPT
from prompts import APPLY_RULES_PROMPT
from test_runner import run_tests


TEMPERATURE_APPLY_RULES = 0.0


# suffix appended to every code generation prompt to force complete, importable solutions
_CODE_GEN_SUFFIX = (
    "\n\nWrite a complete, fully implemented Python solution. "
    "Do NOT use `pass`, empty function bodies, or placeholder docstrings — every function must contain real logic. "
    "Always include ALL necessary imports at the top (e.g. from functools import lru_cache, "
    "from collections import defaultdict, import heapq, etc.). "
    "Wrap your solution in a ```python ... ``` code block."
)


# splits the "APPLIED_RULES: 1,3,4" first line from the improved prompt body
# returns ("1,3,4", "improved prompt text") or ("?", raw) if the line is missing
def _parse_applied_rules(raw: str) -> tuple[str, str]:
    if not raw:
        return "?", raw
    first_line, _, rest = raw.partition('\n')
    first_line = first_line.strip()
    # valid if it contains only digits, commas, and spaces (e.g. "1,3,4")
    if re.match(r'^[\d,\s]+$', first_line) and first_line:
        return first_line.replace(' ', ''), rest.strip()
    return "?", raw


# checks if the llm accidentally inserted python code inside the improved prompt
def _check_answer_leakage(original: str, improved: str) -> bool:
    leak_patterns = [
        r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?\s*:\s*\n\s+',  # python function with indented body
        r'class\s+Solution\s*:',                                  # leetcode solution class
        r'```python\n\s*def\s+',                                  # markdown python block starting with a function
    ]
    for pattern in leak_patterns:
        if re.search(pattern, improved):
            return True
    return False


# get the original prompt from the problem dict
def _build_full_prompt(problem: dict) -> str:
    return problem.get("question_content", "") or problem.get("original_prompt", "")


# packages all evaluation fields into a single result dict
def _make_result(p: dict, original: str, improved: str, code: str,
                 pass_at_1: bool, status: str, tests_passed: int, n_tests: int,
                 applied_rules: str = "?") -> dict:
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
        "applied_rules":      applied_rules,
    }


# applies the transformation rules to one prompt via llm, returns (applied_rules, improved_text)
def apply_transformation_rules(rules: str, original_prompt: str) -> tuple[str, str]:
    prompt = APPLY_RULES_PROMPT.format(rules=rules, original_prompt=original_prompt)
    raw = safe_call(prompt, temperature=TEMPERATURE_APPLY_RULES)
    return _parse_applied_rules(raw)


# generates python code from a prompt, attaching starter code when the problem provides one
def generate_code(prompt: str, starter_code: str = "",
                  temperature: float = GEN_TEMPERATURE) -> str:
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


# evaluates one problem with given rules: transform → check leakage → generate code → run tests (cached)
def evaluate_one(p: dict, rules: str) -> dict:
    # return cached result if already computed for this (problem, rules) pair
    cached = cache_get(p["task_id"], rules)
    if cached is not None and isinstance(cached, dict) and "success" in cached:
        if "canonical_solution" not in cached:
            cached["canonical_solution"] = p.get("canonical_solution", "")
        return cached

    original = _build_full_prompt(p)
    applied_rules, improved_prompt = apply_transformation_rules(rules, original)

    # if the llm returned nothing, fail immediately without evaluating
    if not improved_prompt:
        result = _make_result(p, original, "", "", False,
                              "Empty improved prompt returned by transformation LLM", 0, 0,
                              applied_rules=applied_rules)
        cache_set(p["task_id"], rules, result)
        return result

    # if the llm put actual code in the prompt, discard the result immediately
    if _check_answer_leakage(original, improved_prompt):
        result = _make_result(p, original, improved_prompt, "", False,
                              "Answer leakage detected in transformation", 0, 0,
                              applied_rules=applied_rules)
        cache_set(p["task_id"], rules, result)
        return result

    starter = p.get("starter_code", "") or ""
    code = generate_code(improved_prompt, starter_code=starter)
    pass_at_1, status, tests_passed, n_tests = run_tests(code, p)

    result = _make_result(p, original, improved_prompt, code,
                          pass_at_1, status, tests_passed, n_tests,
                          applied_rules=applied_rules)
    cache_set(p["task_id"], rules, result)
    return result


# evaluates one problem with no transformation (original prompt only), cached under BASELINE_KEY
def evaluate_baseline(p: dict) -> dict:
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


# evaluates a full batch with given rules using 2 batched gpu passes + parallel test execution
def evaluate_batch(problems: list[dict], rules: str) -> list[dict]:
    results = [None] * len(problems)
    uncached_idx = []

    # collect which problems are not yet cached
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

    # gpu pass 1: apply transformation rules to all uncached prompts at once
    t_prompts = [APPLY_RULES_PROMPT.format(rules=rules, original_prompt=o) for o in originals]
    improved_raw = batch_call_llm(t_prompts, temperature=TEMPERATURE_APPLY_RULES)

    # parse the APPLIED_RULES: line from each llm output
    # if the improved prompt is empty, store "" — it will be caught as a fail in _run
    applied_rules_list = []
    improved = []
    for raw in improved_raw:
        ar, imp = _parse_applied_rules(raw if raw else "")
        applied_rules_list.append(ar)
        improved.append(imp.strip() if imp else "")

    # gpu pass 2: generate code from improved prompts (skip empty or leaked)
    leaked = [_check_answer_leakage(o, i) for o, i in zip(originals, improved)]
    c_prompt_idx, c_prompts = [], []
    for j, (p, imp, lk) in enumerate(zip(probs, improved, leaked)):
        if imp and not lk:
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

    # cpu phase: run tests for all problems in parallel
    def _run(j):
        p = probs[j]
        orig, imp, code, lk = originals[j], improved[j], codes[j], leaked[j]
        ar = applied_rules_list[j]
        if not imp:
            # llm returned an empty improved prompt — skip evaluation, count as fail
            r = _make_result(p, orig, imp, "", False,
                             "Empty improved prompt returned by transformation LLM", 0, 0,
                             applied_rules=ar)
        elif lk:
            r = _make_result(p, orig, imp, "", False,
                             "Answer leakage detected in transformation", 0, 0,
                             applied_rules=ar)
        else:
            pass_at_1, status, tp, nt = run_tests(code, p)
            r = _make_result(p, orig, imp, code, pass_at_1, status, tp, nt,
                             applied_rules=ar)
        cache_set(p["task_id"], rules, r)
        return j, r

    with ThreadPoolExecutor(max_workers=min(len(probs), 16)) as ex:
        for j, r in ex.map(_run, range(len(probs))):
            results[uncached_idx[j]] = r

    return results


# evaluates a full batch with no transformation (baseline) using 1 batched gpu pass + parallel tests
def evaluate_baseline_batch(problems: list[dict]) -> list[dict]:
    results = [None] * len(problems)
    uncached_idx = []

    # collect which problems are not yet cached
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

    # gpu pass: generate code from original prompts (no transformation)
    c_prompts = []
    for p, orig in zip(probs, originals):
        starter = p.get("starter_code", "") or ""
        fp = (f"{orig}\n\nComplete this starter code:\n```python\n{starter}\n```{_CODE_GEN_SUFFIX}"
              if starter.strip() else orig + _CODE_GEN_SUFFIX)
        c_prompts.append(fp)

    raw_codes = batch_call_llm(c_prompts, temperature=GEN_TEMPERATURE,
                               system_prompt=CODE_GEN_SYSTEM_PROMPT)
    codes = [extract_code(r) for r in raw_codes]

    # cpu phase: run tests in parallel
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


# convenience wrapper: evaluates rules over problems and returns (pass_count, results)
def score_transformation_rules(rules: str, problems: list) -> tuple[int, list[dict]]:
    results = evaluate_batch(problems, rules)
    success_count = sum(1 for r in results if r["success"])
    return success_count, results
