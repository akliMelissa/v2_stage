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

from config import GEN_TEMPERATURE, MAX_NEW_TOKENS
from cache import cache_get, cache_set, BASELINE_KEY
from code_extraction import extract_code
from model import safe_call
from prompts import APPLY_RULES_PROMPT
from test_runner import run_tests


# ── Answer leakage detection ───────────────────────────────────────────────

def _check_answer_leakage(original: str, improved: str) -> bool:
    """Flag if transformation added code blocks or function definitions.
    
    The mutator sometimes tries to "help" by embedding code snippets
    into the problem description. This catches it.
    
    Returns True if leakage detected, False otherwise.
    """
    # If prompt grew by less than 3x, it's just clarification
    if len(improved) < len(original) * 3:
        return False
    
    new_content = improved[len(original):]
    
    # Look for patterns that indicate actual code was added, not just words
    leak_patterns = [
        r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?\s*:\s*\n\s+\w',  # def with body
        r'class\s+Solution\s*:',                                   # Solution class
        r'\nreturn\s+\w+\(.*?\)',                                  # return func(...)
        r'\n\s*for\s+\w+\s+in\s+.+:\s*\n\s+',                      # for body
        r'```python\n.*?\n```',                                    # code block
    ]
    
    for pattern in leak_patterns:
        if re.search(pattern, new_content, re.DOTALL):
            return True
    
    return False


# ── Prompt transformation ──────────────────────────────────────────────────

def apply_transformation_rules(rules: str, original_prompt: str) -> str:
    """Run the rules over the original prompt and return the improved version."""
    prompt = APPLY_RULES_PROMPT.format(rules=rules, original_prompt=original_prompt)
    improved = safe_call(prompt, temperature=0.3, max_new_tokens=1200)
    
    # Strip any code fences the LLM might have wrapped it in
    improved = re.sub(r"```.*?```", "", improved, flags=re.DOTALL).strip()
    
    # If result is empty or too short, something went wrong — fall back
    if not improved or len(improved) < 20:
        return original_prompt
    
    return improved


# ── Code generation ────────────────────────────────────────────────────────

def generate_code(prompt: str, starter_code: str = "",
                  temperature: float = GEN_TEMPERATURE) -> str:
    """Generate code from a prompt, attaching starter code when present."""
    if starter_code and starter_code.strip():
        full_prompt = (
            f"{prompt}\n\n"
            f"Complete this starter code:\n```python\n{starter_code}\n```"
        )
    else:
        full_prompt = prompt
    
    return extract_code(
        safe_call(full_prompt, temperature=temperature, max_new_tokens=MAX_NEW_TOKENS)
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
        "success":       pass_at_1,
        "error":         "" if pass_at_1 else status,
        "Pass@1":        pass_at_1,
        "Tests_Passed":  tests_passed,
        "n_Tests":       n_tests,
        "Eval_Status":   status,
        "prompt":        original,
        "improved":      improved,
        "code":          code,
        "task_id":       p["task_id"],
    }


def evaluate_one(p: dict, rules: str) -> dict:
    """Evaluate one problem under one rule set. Cached. Returns a result dict.
    
    NEW: Includes answer leakage check after transformation.
    """
    cached = cache_get(p["task_id"], rules)
    if cached is not None and isinstance(cached, dict) and "success" in cached:
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
        return cached

    original = _build_full_prompt(p)
    starter = p.get("starter_code", "") or ""
    code = generate_code(original, starter_code=starter, temperature=GEN_TEMPERATURE)
    pass_at_1, status, tests_passed, n_tests = run_tests(code, p)

    result = _make_result(p, original, original, code,
                          pass_at_1, status, tests_passed, n_tests)
    cache_set(p["task_id"], BASELINE_KEY, result)
    return result


def score_transformation_rules(rules: str, problems: list) -> tuple[int, list[dict]]:
    """Run a rule set over a list of problems. Returns (pass_count, results)."""
    results = [evaluate_one(p, rules) for p in problems]
    success_count = sum(1 for r in results if r["success"])
    return success_count, results