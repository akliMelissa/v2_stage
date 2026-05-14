"""
test_runner.py — Code evaluation harness.

ALIGNED WITH eval_benchmark.py PATTERN:

eval_benchmark.py's approach to function name mismatch:
  1. Build a check(candidate) function that returns (passed, total).
  2. Execute user's code in a clean dict: env = {}; exec(code, env)
  3. Try to find the callable in env by name (exact match, then case-insensitive).
  4. If found, bind it to 'candidate' and call check(candidate).
  5. If NOT found, return (0, "Function `name` not found").
  6. If entry_point is None, try ALL callables in env and pick the best score.

This avoids hardcoding class/method names. It's pure duck typing:
  - If the user named their function differently, it fails gracefully.
  - If the user defined multiple callables, it tries each (greedy).
  - The check function is name-agnostic: it just calls check(whatever_callable).

For LCB (which mixes LeetCode-style + stdin):
  - Functional: build check() that calls candidate as a function.
  - Stdin: build check() that execs candidate as source code.

No more resolve_entry_point() detection. No more class/method inference.
Just pass the function/code and let check(candidate) handle it.
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import tempfile
import zlib
from typing import Any, Dict, List, Optional, Tuple

from config import EVAL_TIMEOUT

# ── SIGALRM handler (Unix-only, best-effort) ──────────────────────────────

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Timeout!")

try:
    signal.signal(signal.SIGALRM, _timeout_handler)
except (AttributeError, ValueError):
    # Windows / non-main-thread: SIGALRM not available. 
    # Real timeout is multiprocessing.Process.join(timeout=...) below.
    pass


# ── Standard imports for all test scripts ──────────────────────────────────

HEADER = (
    "import math\n"
    "import itertools\n"
    "import collections\n"
    "import functools\n"
    "import heapq\n"
    "import bisect\n"
    "import re\n"
    "import sys\n"
    "import json\n"
    "from typing import *\n"
    "from collections import *\n"
    "from math import *\n"
)


# ── Decode test cases (LCB format) ─────────────────────────────────────────

def decode_lcb_tests(raw) -> List[Dict[str, str]]:
    """LCB ships test cases as: list | JSON string | base64+zlib+JSON."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        decoded = zlib.decompress(base64.b64decode(raw.encode("utf-8"))).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return []


# ── Build check(candidate) for FUNCTIONAL (LeetCode-style) problems ────────

def _build_functional_check(tests: List[Dict[str, str]]) -> Tuple[str, int]:
    """Build check(candidate) → (passed, total) for functional problems.
    
    The check function receives candidate as a callable.
    It doesn't care about the function's NAME — it just calls it.
    
    This mirrors eval_benchmark.py's approach: the check harness is
    name-agnostic. The actual function is found by _safe_exec().
    """
    total = len(tests)
    tests_json = json.dumps(tests)
    
    body = f"""
import ast, json
def _safe_parse(s):
    s = s.strip()
    if not s:
        return s
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return s

def _normalize(x):
    if isinstance(x, tuple):
        return [_normalize(v) for v in x]
    if isinstance(x, list):
        return [_normalize(v) for v in x]
    if isinstance(x, dict):
        return {{k: _normalize(v) for k, v in x.items()}}
    return x

def check(candidate):
    \"\"\"Test harness: candidate is the actual function/method to call.
    
    Note: We don't care about the function's name. We just call it.
    \"\"\"
    _TESTS = {tests_json}
    passed = 0
    total  = {total}
    for _tc in _TESTS:
        _inp_raw = _tc.get('input', '')
        _exp_raw = _tc.get('output', '')
        # Parse input arguments (newline-separated)
        _args = [_safe_parse(_line) for _line in str(_inp_raw).strip().split(chr(10)) if _line.strip()]
        _exp = _safe_parse(_exp_raw)
        try:
            _got = candidate(*_args)
        except Exception:
            continue
        if _normalize(_got) == _normalize(_exp):
            passed += 1
    return passed, total
"""
    return body, total


# ── Build check(candidate) for STDIN/STDOUT (Codeforces-style) problems ────

def _build_stdin_check(tests: List[Dict[str, str]]) -> Tuple[str, int]:
    """Build check(candidate) → (passed, total) for stdin/stdout problems.
    
    The check function receives candidate as raw source code (string).
    It execs the code once per test, patching stdin/stdout.
    
    This handles problems like Codeforces where the solution reads from
    stdin and prints to stdout, with no function signature constraint.
    """
    total = len(tests)
    tests_json = json.dumps(tests)
    
    body = f"""
import sys, io
def check(candidate):
    \"\"\"Test harness: candidate is the raw source code (string).\"\"\"
    _TESTS = {tests_json}
    passed = 0
    total  = {total}
    for _tc in _TESTS:
        _inp = _tc.get('input', '')
        _exp = _tc.get('output', '').strip()
        # Patch stdin
        sys.stdin = io.StringIO(_inp)
        _out_buf = io.StringIO()
        _old_stdout = sys.stdout
        sys.stdout = _out_buf
        try:
            exec(candidate, {{'__name__': '__main__'}})
        except SystemExit:
            pass
        except Exception:
            sys.stdout = _old_stdout
            continue
        sys.stdout = _old_stdout
        _got = _out_buf.getvalue().strip()
        if _got == _exp:
            passed += 1
    return passed, total
"""
    return body, total


# ── Sandbox execution (eval_benchmark.py style) ────────────────────────────

def _safe_exec(
    candidate_code: str,
    check_code: str,
    queue: mp.Queue,
    entry_point: Optional[str] = None,
    stdin_mode: bool = False,
) -> None:
    """Execute code safely in a subprocess.
    
    FUNCTION NAME RESOLUTION (eval_benchmark.py pattern):
    
    If entry_point is given (e.g., "solution" from starter code):
      1. Try exact match: env.get("solution")
      2. Try case-insensitive: look for "solution" (any case)
      3. If still not found: fail with "Function `solution` not found"
      4. Set fns = [the_function]
    
    If entry_point is None (no starter code, or we don't know the name):
      1. Extract ALL callables from env
      2. Try each one, return the best score
      3. If no callables found: fail with "Function not found"
    
    Then for each candidate function:
      - Bind it to env["candidate"]
      - Exec check_code with that binding
      - Extract (passed, total) from env["_result"]
      - Return the best score (max if multiple callables)
    
    For stdin_mode, candidate_code is the raw source, not a function.
    """
    try:
        env: Dict[str, Any] = {}
        exec(candidate_code, env)
        
        if not stdin_mode:
            # Functional mode: find the callable by name
            if entry_point:
                fn = env.get(entry_point)
                if not callable(fn):
                    # Try case-insensitive match
                    for k, v in env.items():
                        if k.lower() == entry_point.lower() and callable(v):
                            fn = v
                            break
                if not callable(fn):
                    queue.put((0, f"Function `{entry_point}` not found"))
                    return
                fns = [fn]
            else:
                # No entry point: try ALL callables
                fns = [v for v in env.values() if callable(v)]
            
            if not fns:
                queue.put((0, "Function not found"))
                return
        else:
            # Stdin mode: candidate_code is the source string
            # We'll pass it directly to check() as a string
            fns = [candidate_code]
        
        def run(fn) -> int:
            env["candidate"] = fn
            exec(check_code + "\n_result = check(candidate)", env)
            passed, _ = env["_result"]
            return passed
        
        best_score = max(run(f) for f in fns)
        queue.put((best_score, "OK"))
    
    except Exception as exc:
        queue.put((0, f"ERROR: {type(exc).__name__}: {exc}"))


def evaluate_with_timeout(
    candidate_code: str,
    check_code: str,
    *,
    timeout_seconds: int = 20,
    entry_point: Optional[str] = None,
    stdin_mode: bool = False,
) -> Tuple[int, str]:
    """Run code + check in a subprocess with a wall-clock timeout.
    
    Spawns a Process, waits up to timeout_seconds, kills if needed.
    Returns (tests_passed, status_message).
    
    Status message is "OK" on success, "ERROR: ..." on failure.
    """
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_safe_exec,
        args=(candidate_code, check_code, queue, entry_point, stdin_mode)
    )
    proc.start()
    proc.join(timeout=timeout_seconds)
    
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return 0, "ERROR: Timeout/Killed"
    
    try:
        return queue.get_nowait()
    except Exception:
        return 0, "ERROR: Unknown"


# ── Public entry point ─────────────────────────────────────────────────────

def run_tests(code: str, problem: dict) -> Tuple[bool, str, int, int]:
    """Run all tests for an LCB problem.
    
    Returns (Pass@1, Eval_Status, Tests_Passed, n_Tests).
    
    Aligned with eval_benchmark.py: 
      - Builds a check(candidate) harness
      - Runs it via _safe_exec in a subprocess
      - Handles function name resolution gracefully
      - Returns (success_bool, status_string, passed_count, total_count)
    """
    if not code.strip():
        return False, "Empty code generated", 0, 0

    public_tests  = decode_lcb_tests(problem.get("public_test_cases",  "[]"))
    private_tests = decode_lcb_tests(problem.get("private_test_cases", "[]"))
    tests = (public_tests or []) + (private_tests or [])
    if not tests:
        return False, "No test cases available", 0, 0

    starter_code = problem.get("starter_code", "") or ""
    is_functional = bool(starter_code.strip())

    if is_functional:
        # Functional mode: extract the expected function name from starter code
        m = re.search(r"def\s+(\w+)\s*\(", starter_code)
        entry_point = m.group(1) if m else None
        
        check_code, n_tests = _build_functional_check(tests)
        
        # Try with the entry point hint. If it fails, _safe_exec will report it.
        passed, status = evaluate_with_timeout(
            code, check_code, timeout_seconds=EVAL_TIMEOUT,
            entry_point=entry_point, stdin_mode=False
        )
    else:
        # Stdin mode: code is raw source, not a function
        check_code, n_tests = _build_stdin_check(tests)
        passed, status = evaluate_with_timeout(
            code, check_code, timeout_seconds=EVAL_TIMEOUT,
            entry_point=None, stdin_mode=True
        )

    pass_at_1 = (passed == n_tests) and (status == "OK")
    return pass_at_1, status, passed, n_tests