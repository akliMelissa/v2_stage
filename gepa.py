"""
gepa.py — The GEPA optimization loop.

Each generation:
  1. Sample a minibatch from the dev split.
  2. Score the current rules on the batch.
  3. If anything failed, ask the LLM to mutate the rules based on the failures.
  4. Re-score the mutated rules. Accept if strictly better on the batch.
  5. Update the Pareto tracker on the held-out val split.

At the end, return the best rules under (max val score, then unique-coverage tiebreak).
"""

import random

from config import GENERATIONS, MINIBATCH_SIZE, VAL_SIZE
from evaluator import evaluate_one, score_transformation_rules
from model import safe_call
from prompts import INITIAL_TRANSFORMATION_RULES, MUTATE_RULES_PROMPT


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate_transformation_rules(current_rules: str, failures_report: str,
                                batch_size: int, passed: int) -> str:
    """Improve transformation rules from full failure traces."""
    prompt = MUTATE_RULES_PROMPT.format(
        current_rules=current_rules,
        batch_size=batch_size,
        passed=passed,
        failures=failures_report,
    )
    new_rules = safe_call(prompt, temperature=0.7, max_new_tokens=1500)
    if not new_rules or len(new_rules) < 50:
        print("  [WARNING] Mutation produced empty/short output, keeping current rules")
        return current_rules
    return new_rules


# ── Pareto tracker ────────────────────────────────────────────────────────────

class Pareto:
    """Tracks val-set scores for every rule set we've tested.

    `top()` picks the rules with the highest score, breaking ties by
    unique problem coverage (rules that pass problems no one else does).

    ENHANCED: Maintains instance-level Pareto frontier for weighted selection.
    - `_instance_wins[rules]` = set of problem indices where this rule succeeds
    - `best_per_instance[idx]` = best rule for problem idx
    - `select_for_mutation_weighted()` = select parent weighted by instance wins
    """

    def __init__(self, val: list):
        self.val = val
        self._scores: dict = {}
        self._per_problem: dict = {}
        self._seen: set = set()
        # ← NEW LINE 59-60: Instance-level Pareto frontier tracking
        self._instance_wins: dict = {}  # ← NEW: rules -> set of winning problem indices
        self.best_per_instance: dict = {}  # ← NEW: problem idx -> best rule

    def update(self, rules: str, gen: int) -> int:
        if rules in self._seen:
            return self._scores.get(rules, -1)
        self._seen.add(rules)

        score, per = score_transformation_rules(rules, self.val)
        self._scores[rules] = score
        self._per_problem[rules] = [r["success"] for r in per]

        # ← NEW SECTION 71-84: Track instance wins
        self._instance_wins[rules] = set()
        for idx, r in enumerate(per):
            if r["success"]:
                self._instance_wins[rules].add(idx)

                # Update best_per_instance
                current_best = self.best_per_instance.get(idx)
                if current_best is None:
                    self.best_per_instance[idx] = rules
                else:
                    # Check if new rule is better for this instance
                    if self._per_problem[rules][idx] and not self._per_problem.get(current_best, [False] * len(per))[idx]:
                        self.best_per_instance[idx] = rules

        for i, r in enumerate(per, 1):
            status = "PASS" if r["success"] else "FAIL"
            print(f"  [gen{gen} pareto] {i}/{len(per)} — {status}")
        # ← CHANGED LINE 89: Now shows instance wins count
        print(f"  [gen{gen} pareto] total: {score}/{len(per)}, instance wins: {len(self._instance_wins[rules])}")
        return score

    # ← CHANGED SECTION 92-102: Simplified top() using instance wins
    def top(self) -> str:
        if not self._scores:
            return INITIAL_TRANSFORMATION_RULES
        max_score = max(self._scores.values())
        candidates = [r for r, s in self._scores.items() if s == max_score]
        if len(candidates) == 1:
            return candidates[0]

        # ← CHANGED: Use instance wins as tiebreaker (aligned with GEPA)
        candidate_wins = {r: len(self._instance_wins.get(r, set())) for r in candidates}
        return max(candidate_wins, key=candidate_wins.get)

    # ← NEW METHOD 104-135: Weighted parent selection (CORE GEPA MECHANISM)
    def select_for_mutation_weighted(self) -> str:
        """Select a rule to mutate, weighted by how many problems it solves.

        ← NEW: This implements weighted parent selection from GEPA.
        Rules that solve more instances are more likely to be selected for mutation.
        """
        if not self._scores:
            return INITIAL_TRANSFORMATION_RULES

        # Weight rules by their instance wins
        weighted_rules = []
        weights = []

        for rules, wins in self._instance_wins.items():
            if wins:  # Only consider rules that win on at least 1 problem
                weighted_rules.append(rules)
                weights.append(len(wins))

        if not weighted_rules:
            # Fallback: use the top rule
            return self.top()

        # Select weighted by wins
        total = sum(weights)
        choice = random.random() * total
        cumsum = 0
        for r, w in zip(weighted_rules, weights):
            cumsum += w
            if choice <= cumsum:
                return r

        return weighted_rules[-1]  # Safety fallback


# ── Failure report formatting ─────────────────────────────────────────────────

def _format_failures(batch_results: list[dict]) -> str:
    report = ""
    for r in batch_results:
        if not r["success"]:
            report += f"ORIGINAL PROMPT:\n{r['prompt'][:600]}\n\n"
            report += f"IMPROVED PROMPT:\n{r['improved'][:600]}\n\n"
            report += f"CODE GENERATED:\n{r['code']}\n\n"
            report += f"ERROR:\n{r['error']}\n"
            report += "-" * 20 + "\n"
    return report


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_gepa(problems: list) -> tuple[str, list[dict]]:
    """Run the GEPA optimization loop. Returns (best_rules, generation_log)."""
    problems = list(problems)
    random.seed(42)
    random.shuffle(problems)
    val = problems[:VAL_SIZE]
    dev = problems[VAL_SIZE:]

    pareto = Pareto(val)
    current_rules = INITIAL_TRANSFORMATION_RULES
    gen_log = []

    print("GEPA starting...")
    print(f"Initial transformation rules:\n{current_rules}\n")
    print("=" * 80)

    pareto.update(current_rules, gen=-1)

    for i in range(GENERATIONS):
        batch = random.sample(dev, min(MINIBATCH_SIZE, len(dev)))

        # ← NEW SECTION 175-181: Select parent rule weighted by instance wins (CORE CHANGE)
        if i > 0:
            current_rules = pareto.select_for_mutation_weighted()
            instance_wins = len(pareto._instance_wins.get(current_rules, set()))
            print(f"\n── Gen {i} — selected rules with {instance_wins} instance wins ──")
        else:
            print(f"\n── Gen {i} — testing current transformation rules ──")

        batch_results = [evaluate_one(p, current_rules) for p in batch]
        score = sum(1 for r in batch_results if r["success"])
        for j, r in enumerate(batch_results, 1):
            status = "PASS" if r["success"] else f"FAIL ({r['error']})"
            print(f"  [gen{i} eval] {j}/{len(batch)} — {status}")
        print(f"Gen {i} score: {score}/{len(batch)}")

        mutation_accepted = False
        candidate_rules = None
        cand_score = None

        if score < len(batch):
            print(f"Gen {i} — mutating transformation rules...")
            failures_report = _format_failures(batch_results)
            candidate_rules = mutate_transformation_rules(
                current_rules, failures_report, len(batch), score
            )

            if candidate_rules and candidate_rules != current_rules:
                print(f"Gen {i} — testing mutated rules...")
                cand_results = [evaluate_one(p, candidate_rules) for p in batch]
                cand_score = sum(1 for r in cand_results if r["success"])
                for j, r in enumerate(cand_results, 1):
                    status = "PASS" if r["success"] else f"FAIL ({r['error']})"
                    print(f"  [gen{i} cand] {j}/{len(batch)} — {status}")
                print(f"Gen {i} candidate score: {cand_score}/{len(batch)}")

                if cand_score > score:
                    current_rules = candidate_rules
                    mutation_accepted = True
                    print(f"Gen {i} — mutation accepted.")
                    print(f"\nNew transformation rules:\n{current_rules}\n")
                    print("=" * 80)
                else:
                    print(f"Gen {i} — mutation rejected (worse score).")
            else:
                print(f"Gen {i} — mutation failed or unchanged.")
        else:
            print(f"Gen {i} — perfect score on batch, skipping mutation.")

        pareto.update(current_rules, i)

        # ← NEW LINE 234: Track instance wins in generation log
        gen_log.append({
            "gen":               i,
            "rules":             current_rules,
            "batch":             [r["task_id"] for r in batch_results],
            "score":             score,
            "batch_size":        len(batch),
            "mutation_accepted": mutation_accepted,
            "candidate_rules":   candidate_rules,
            "candidate_score":   cand_score,
            "instance_wins":     len(pareto._instance_wins.get(current_rules, set())),  # ← NEW
            "details": [
                {
                    "task_id":  r["task_id"],
                    "success":  r["success"],
                    "error":    r["error"],
                    "prompt":   r["prompt"],
                    "improved": r["improved"],
                    "code":     r["code"],
                }
                for r in batch_results
            ],
        })

    best_rules = pareto.top()
    print(f"\n{'='*80}")
    print("BEST TRANSFORMATION RULES (highest val score, unique-coverage tiebreaker):")
    print(f"{'='*80}")
    print(best_rules)
    print(f"{'='*80}\n")
    return best_rules, gen_log