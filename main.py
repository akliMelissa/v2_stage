"""
main.py — Entry point.

Runs:
  1. Load LiveCodeBench problems.
  2. Baseline evaluation (no transformation, cached).
  3. GEPA optimization loop.
  4. Final evaluation with the best rules.
  5. Save results.json and gen_log.json.

To swap benchmark:        edit data_loader.py and config.py (LCB_* constants).
To swap model:            edit MODEL_NAME in config.py.
To tune the GEPA loop:    edit NUM_PROBLEMS / GENERATIONS / MINIBATCH_SIZE / VAL_SIZE in config.py.
"""

import json

from config import NUM_PROBLEMS, RESULTS_DIR
from data_loader import load_livecodebench
from evaluator import evaluate_baseline, evaluate_one
from gepa import run_gepa


def final_eval(rules: str, problems: list) -> list[dict]:
    """Evaluate final rules on every problem. Returns list of result dicts."""
    results = []
    for i, p in enumerate(problems, 1):
        r = evaluate_one(p, rules)
        results.append(r)
        status = "PASS" if r["success"] else f"FAIL ({r['error']})"
        print(f"  [final] {i}/{len(problems)} — {status}", flush=True)
    return results


def main():

    # ── Load ──────────────────────────────────────────────────────────────────
    problems = load_livecodebench(NUM_PROBLEMS)
    print(f"Loaded {len(problems)} LiveCodeBench problems")
    print("=" * 80)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"\nComputing baseline on {len(problems)} problems (cached after first run)...")
    
    baseline_details = []
    for i, p in enumerate(problems, 1):
        r = evaluate_baseline(p)
        baseline_details.append(r)
        status = "PASS" if r["success"] else f"FAIL ({(r['error'] or '')[:60]})"
        print(f"  [baseline] {i}/{len(problems)} — {status}", flush=True)
    baseline_score = sum(r["success"] for r in baseline_details)
    print(
        f"\nBaseline (no transformation): {baseline_score}/{len(baseline_details)} "
        f"({baseline_score/len(problems)*100:.1f}%)"
    )

    # ── GEPA ──────────────────────────────────────────────────────────────────
    best_rules, gen_log = run_gepa(problems)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nRunning final evaluation with best transformation rules...")
    gepa_details = final_eval(best_rules, problems)
    gepa_score = sum(r["success"] for r in gepa_details)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FINAL RESULTS — LiveCodeBench")
    print(f"{'='*80}")
    print(
        f"Baseline : {baseline_score}/{len(baseline_details)} "
        f"({baseline_score/len(problems)*100:.1f}%)"
    )
    print(
        f"GEPA     : {gepa_score}/{len(gepa_details)} "
        f"({gepa_score/len(problems)*100:.1f}%)"
    )
    delta = gepa_score - baseline_score
    print(f"Delta    : {delta:+d} ({delta/len(problems)*100:+.1f}%)")
    print(f"{'='*80}")

    # ── Per-problem analysis ──────────────────────────────────────────────────
    per_problem = []
    for p, b, g in zip(problems, baseline_details, gepa_details):
        b_ok, g_ok = b["success"], g["success"]
        if not b_ok and g_ok:
            status = "improved"
        elif b_ok and not g_ok:
            status = "regressed"
        elif b_ok and g_ok:
            status = "stable_pass"
        else:
            status = "stable_fail"
        per_problem.append({
            "task_id":         p["task_id"],
            "status":          status,
            "baseline_pass":   b_ok,
            "gepa_pass":       g_ok,
            "baseline_error":  b["error"],
            "gepa_error":      g["error"],
            "original_prompt": b["prompt"],
            "improved_prompt": g.get("improved", ""),
            "baseline_code":   b["code"],
            "gepa_code":       g["code"],
        })

    improved  = [x for x in per_problem if x["status"] == "improved"]
    regressed = [x for x in per_problem if x["status"] == "regressed"]
    print(f"\nImproved  (+): {len(improved)}  — {[x['task_id'] for x in improved]}")
    print(f"Regressed (-): {len(regressed)} — {[x['task_id'] for x in regressed]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump({
            "benchmark":                 "LiveCodeBench",
            "num_problems":              len(problems),
            "baseline_score":            baseline_score,
            "gepa_score":                gepa_score,
            "delta":                     delta,
            "best_transformation_rules": best_rules,
            "per_problem":               per_problem,
        }, f, indent=2)

    with open(RESULTS_DIR / "gen_log.json", "w") as f:
        json.dump(gen_log, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/results.json")
    print(f"Generation log saved to {RESULTS_DIR}/gen_log.json")


if __name__ == "__main__":
    main()
