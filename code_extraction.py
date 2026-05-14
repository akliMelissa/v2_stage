"""
code_extraction.py — Extract Python code from LLM output.

The test harness (test_runner.py via _safe_exec) now handles function
name resolution, so we only need code extraction here.

This is eval_benchmark.py's 3-tier extraction approach.
"""

import re
import textwrap


def extract_code(txt: str) -> str:
    """Extract Python code from LLM output.
    
    Tries in order:
    1. Markdown ```python ... ``` fence (handles most models)
    2. [PYTHON] ... [/PYTHON] tags (some models use this)
    3. First import/def/class block (model skipped fence)
    4. Fallback: dedent the raw text
    
    This approach is from eval_benchmark.py and handles model
    output variation well.
    """
    # 1. Markdown fence (most common)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    
    # 2. [PYTHON] tags (less common, some models)
    m = re.search(r"\[PYTHON\](.*?)\[/PYTHON\]", txt, re.DOTALL | re.IGNORECASE)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    
    # 3. First import/from/def/class block (model skipped fence)
    #    This catches cases like: "Here's a solution:\nimport sys\ndef foo():"
    m = re.search(
        r"^((?:(?:import|from)\s+\S[^\n]*\n)*[ \t]*def\s+\w+\s*\(.*)",
        txt,
        re.MULTILINE | re.DOTALL,
    )
    if m:
        return textwrap.dedent(m.group(1)).strip()
    
    # 4. Fallback: just dedent and strip
    return textwrap.dedent(txt).strip()