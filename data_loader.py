"""
data_loader.py — Load LiveCodeBench problems from HuggingFace.

Returns a list of dicts with the fields the rest of the pipeline expects:
    task_id, question_content, original_prompt, starter_code,
    public_test_cases, private_test_cases, difficulty, platform,
    canonical_solution
"""

from datasets import load_dataset

from config import (
    LCB_DATASET,
    LCB_VERSION_TAG,
    LCB_FALLBACK_REPO,
    NUM_PROBLEMS,
)


def load_livecodebench(n: int = NUM_PROBLEMS) -> list[dict]:
    """Load the first `n` LiveCodeBench problems.

    Tries the official livecodebench repo first; falls back to the
    community mirror if the official one is unavailable.
    """
    print("Loading LiveCodeBench from HuggingFace...")
    try:
        ds = load_dataset(
            LCB_DATASET,
            version_tag=LCB_VERSION_TAG,
            split="test",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Falling back to {LCB_FALLBACK_REPO} mirror: {e}")
        ds = load_dataset(LCB_FALLBACK_REPO, LCB_VERSION_TAG, split="test")

    problems = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        problems.append(_normalize_row(row, i))
    return problems


def _normalize_row(row: dict, i: int) -> dict:
    """Shape an LCB dataset row into the dict the rest of the pipeline expects."""
    content = row.get("question_content", "") or ""
    canonical = (
        row.get("canonical_solution", "")
        or row.get("solution", "")
        or row.get("code", "")
        or ""
    )
    return {
        "task_id":            str(row.get("question_id", f"lcb_{i}")),
        "question_content":   content,
        "original_prompt":    content,
        "starter_code":       row.get("starter_code", "") or "",
        "public_test_cases":  row.get("public_test_cases", "[]"),
        "private_test_cases": row.get("private_test_cases", "[]"),
        "difficulty":         row.get("difficulty", "unknown"),
        "platform":           row.get("platform", "unknown"),
        "canonical_solution": canonical,
    }


if __name__ == "__main__":
    problems = load_livecodebench(5)
    for p in problems:
        print(p["task_id"], "—", p["difficulty"], "—", p["platform"])
        print(p["question_content"][:200])
        print()