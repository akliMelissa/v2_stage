"""
main.py — Entry point.

Runs:
  1. Load LiveCodeBench problems.
  2. Baseline evaluation (no transformation, cached).
  3. GEPA optimization loop.
  4. Final evaluation with the best rules.
  5. Save results.json and analysis.md.

change benchmark:        edit data_loader.py and config.py (LCB_* constants).
change model:            edit MODEL_NAME in config.py.
change configs of the GEPA loop:    edit NUM_PROBLEMS / GENERATIONS / MINIBATCH_SIZE / VAL_SIZE in config.py.
"""

import json
from datetime import datetime

from config import (
    NUM_PROBLEMS, RESULTS_DIR, MODEL_NAME,
    GENERATIONS, MINIBATCH_SIZE, VAL_SIZE, EVAL_TIMEOUT,
    LCB_DATASET, LCB_VERSION_TAG, GEN_TEMPERATURE, POPULATION_SIZE, USE_MERGE,
)
from data_loader import load_livecodebench
from evaluator import evaluate_baseline_batch, evaluate_batch
from model import safe_call
from prompts import INITIAL_TRANSFORMATION_RULES
from gepa import run_gepa


# evaluates the best rules found by gepa on every problem and prints results
def final_eval(rules: str, problems: list) -> list[dict]:
    results = evaluate_batch(problems, rules)
    for i, r in enumerate(results, 1):
        status = "PASS" if r["success"] else f"FAIL ({r['error']})"
        print(f"  [final] {i}/{len(problems)} — {status}", flush=True)
    return results


# computes baseline scores on all problems, cached after first run
def run_baseline(problems: list) -> tuple[list[dict], int]:
    print(f"\nComputing baseline on {len(problems)} problems (cached after first run)...")
    baseline_details = evaluate_baseline_batch(problems)
    for i, r in enumerate(baseline_details, 1):
        status = "PASS" if r["success"] else f"FAIL ({(r['error'] or '')[:60]})"
        print(f"  [baseline] {i}/{len(problems)} — {status}", flush=True)
    baseline_score = sum(r["success"] for r in baseline_details)
    print(
        f"\nBaseline (no transformation): {baseline_score}/{len(baseline_details)} "
        f"({baseline_score/len(problems)*100:.1f}%)"
    )
    return baseline_details, baseline_score


# runs the gepa loop to find the best transformation rules
def run_optimization(problems: list, baseline_details: list) -> str:
    best_rules, _ = run_gepa(
        problems,
        initial_rules=INITIAL_TRANSFORMATION_RULES,
        evaluate_fn=lambda rules, examples: evaluate_batch(examples, rules),
        llm_reflect_fn=safe_call,
        llm_merge_fn=safe_call if USE_MERGE else None,
        generations=GENERATIONS,
        minibatch_size=MINIBATCH_SIZE,
        val_size=VAL_SIZE,
        population_size=POPULATION_SIZE,
        baseline_results=baseline_details,
    )
    return best_rules


# prints the final summary table (baseline vs gepa scores and delta)
def print_summary(problems, baseline_details, baseline_score, gepa_details, gepa_score):
    delta = gepa_score - baseline_score
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
    print(f"Delta    : {delta:+d} ({delta/len(problems)*100:+.1f}%)")
    print(f"{'='*80}")
    return delta


# builds the per-problem comparison list (improved / regressed / stable_pass / stable_fail)
def build_per_problem(problems, baseline_details, gepa_details) -> list[dict]:
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
    return per_problem


# saves results.json with all scores and per-problem details
def save_results(problems, baseline_score, gepa_score, delta, best_rules, per_problem):
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
    print(f"\nResults saved to {RESULTS_DIR}/results.json")


# generates and saves the markdown analysis report (analysis.md)
def save_analysis(problems, baseline_score, gepa_score, delta, best_rules, per_problem):
    improved    = [x for x in per_problem if x["status"] == "improved"]
    regressed   = [x for x in per_problem if x["status"] == "regressed"]
    stable_pass = [x for x in per_problem if x["status"] == "stable_pass"]
    stable_fail = [x for x in per_problem if x["status"] == "stable_fail"]

    report_lines = [
        f"# GEPA Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Model & Benchmark",
        f"- **LLM**: `{MODEL_NAME}`",
        f"- **Benchmark**: {LCB_DATASET} / `{LCB_VERSION_TAG}`",
        f"- **Problems evaluated**: {len(problems)}",
        "",
        "## Configuration",
        f"- Generations: {GENERATIONS}",
        f"- Minibatch size: {MINIBATCH_SIZE}",
        f"- Val size: {VAL_SIZE}",
        f"- Eval timeout: {EVAL_TIMEOUT}s",
        f"- Gen temperature: {GEN_TEMPERATURE}",
        "",
        "## Results",
        f"| | Passed | % |",
        f"|---|---|---|",
        f"| Baseline | {baseline_score}/{len(problems)} | {baseline_score/len(problems)*100:.1f}% |",
        f"| GEPA     | {gepa_score}/{len(problems)} | {gepa_score/len(problems)*100:.1f}% |",
        f"| **Delta** | **{delta:+d}** | **{delta/len(problems)*100:+.1f}%** |",
        "",
        f"| Status | Count |",
        f"|---|---|",
        f"| Improved (+) | {len(improved)} |",
        f"| Regressed (-) | {len(regressed)} |",
        f"| Stable pass | {len(stable_pass)} |",
        f"| Stable fail | {len(stable_fail)} |",
        "",
        "## Initial Transformation Rules",
        "```",
        INITIAL_TRANSFORMATION_RULES.strip(),
        "```",
        "",
        "## Best Transformation Rules (after GEPA)",
        "```",
        best_rules.strip(),
        "```",
        "",
        f"## Improved Problems (+{len(improved)})",
    ]
    for x in improved:
        report_lines.append(f"- `{x['task_id']}`")

    report_lines += [
        "",
        f"## Regressed Problems (-{len(regressed)})",
    ]
    for x in regressed:
        report_lines.append(
            f"- `{x['task_id']}` — baseline error: `{x['baseline_error'] or 'OK'}` "
            f"→ gepa error: `{x['gepa_error']}`"
        )

    report_path = RESULTS_DIR / "analysis.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Analysis report saved to {report_path}")


def main():
    # load the benchmark problems from livecodebench
    problems = load_livecodebench(NUM_PROBLEMS)
    print(f"Loaded {len(problems)} LiveCodeBench problems")
    print("=" * 80)

    # compute baseline (original prompts, no transformation)
    baseline_details, baseline_score = run_baseline(problems)

    # run gepa to find the best transformation rules
    best_rules = run_optimization(problems, baseline_details)

    # evaluate the best rules on all problems
    print("\nRunning final evaluation with best transformation rules...")
    gepa_details = final_eval(best_rules, problems)
    gepa_score = sum(r["success"] for r in gepa_details)

    # print summary and build per-problem comparison
    delta = print_summary(problems, baseline_details, baseline_score, gepa_details, gepa_score)
    per_problem = build_per_problem(problems, baseline_details, gepa_details)

    improved  = [x for x in per_problem if x["status"] == "improved"]
    regressed = [x for x in per_problem if x["status"] == "regressed"]
    print(f"\nImproved  (+): {len(improved)}  — {[x['task_id'] for x in improved]}")
    print(f"Regressed (-): {len(regressed)} — {[x['task_id'] for x in regressed]}")

    # save results and analysis report to disk
    save_results(problems, baseline_score, gepa_score, delta, best_rules, per_problem)
    save_analysis(problems, baseline_score, gepa_score, delta, best_rules, per_problem)


if __name__ == "__main__":
    main()
