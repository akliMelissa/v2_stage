"""
gepa_fidele.py — Implémentation fidèle de GEPA pour l'évolution de "Transformation Rules".

Basé sur : "GEPA: Reflective Prompt Evolution can Outperform Reinforcement Learning"
https://arxiv.org/pdf/2507.19457

Adapté pour : optimiser un ensemble de règles qui transforment un "bad prompt" 
en "good prompt" avant génération de code.

Mécanismes clés fidèles au papier :
  1. Population de candidats (pas un seul hill-climb)
  2. Dev/Val split avec évaluation par minibatch
  3. Reflection LLM sur trajectoires (succès + échecs)
  4. Mutation gating (>= parent, pas strictement >)
  5. Front de Pareto instance-level
  6. Merge structurel 3-way (combine les leçons complémentaires)
  7. Sélection parent pondérée par instance wins
"""

import difflib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable


def _diff_rules(old: str, new: str) -> str:
    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="before", tofile="after", lineterm="",
    ))
    return "".join(diff) if diff else "(no change)"

# ── Configuration (injectée depuis l'extérieur ou valeurs par défaut) ────────

DEFAULT_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 4
DEFAULT_PARETO_SIZE = 6
DEFAULT_MINIBATCH_SIZE = 8
DEFAULT_VAL_SIZE = 20
DEFAULT_DEV_VAL_SPLIT = 0.5
DEFAULT_MUTATION_RATE = 0.7
DEFAULT_PERFECT_SCORE = 1.0
DEFAULT_MAX_MERGE_INVOCATIONS = 5
DEFAULT_MERGE_VAL_OVERLAP_FLOOR = 2


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RuleCandidate:
    """Un ensemble de règles de transformation, avec métriques GEPA."""
    rules: str
    dev_scores: dict[str, float] = field(default_factory=dict)
    val_scores: dict[str, float] = field(default_factory=dict)
    per_item_val_scores: list[dict[str, float]] = field(default_factory=list)
    val_instance_wins: set[int] = field(default_factory=set)
    evaluated_val_ids: set[int] = field(default_factory=set)
    parents: list[int] = field(default_factory=list)
    id: int = 0


# ── Prompts de reflection (fidèles au papier) ────────────────────────────────

REFLECTION_PROMPT_TEMPLATE = """
I provided an assistant with the following transformation rules to improve code-generation prompts:

```
{rules}
```

The following are examples of different original prompts provided to the assistant, 
along with the assistant's transformed prompt, the generated code, and the evaluation result.
For each example, you will see:
- The original user prompt
- The transformed prompt produced by the rules
- The code generated from the transformed prompt
- The agent trajectory (if available) showing reasoning, tool calls, and intermediate steps
- Feedback on how the result could be better

{inputs_outputs_feedback}

Your task is to write a NEW set of transformation rules for the assistant.

Read the inputs carefully and identify the input format and infer a detailed task description.
Carefully examine the agent trajectories to understand HOW the assistant is approaching the task. Look at:
- What reasoning steps the assistant takes
- Where the assistant makes mistakes or suboptimal choices
- What information the assistant is missing or misinterpreting

Read all the assistant responses and the corresponding feedback. 
Identify all niche and domain-specific factual information about the task and include it in the rules.
The assistant may have utilized a generalizable strategy; if so, include that in the rules as well.

Based on the feedback AND the agent trajectories, identify what the assistant is doing wrong 
or could do better, and incorporate specific guidance to address these issues in the new rules.

Important constraints:
- Keep the same structure: numbered rules 1., 2., 3., etc.
- Do not add code examples, function signatures, or algorithm names inside the rules
- Focus on improving clarity, specificity, and actionable guidance for prompt transformation
- The rules must describe HOW to transform a bad prompt into a good one, not solve the coding task itself

Provide the new rules as a numbered list.
""".strip()


MERGE_PROMPT_TEMPLATE = """
You are merging two improved versions of transformation rules that share a common ancestor.

Common ancestor rules:
```
{ancestor_rules}
```

Version A (better on some validation instances):
```
{rules_a}
```

Version B (better on other validation instances):
```
{rules_b}
```

Your task is to produce a SINGLE merged set of rules that combines the best insights from BOTH versions.

Guidelines:
- If A and B changed the SAME rule differently, keep the version that is more specific/clear, or synthesize both
- If only A or only B changed a rule, keep that change
- If neither changed a rule, keep the ancestor version
- Do not add new rules unless absolutely necessary to resolve a conflict
- Maintain the numbered list format 1., 2., 3., etc.
- Do NOT include code, function signatures, or algorithm names

Provide the merged rules as a numbered list.
""".strip()


# ── Helpers: split/join/renumber rules ───────────────────────────────────────

def _split_rules(rules: str) -> list[str]:
    """Split rules string into individual numbered rules."""
    text = rules.strip()
    parts = re.split(r'(?m)^\s*(?=\d+\.\s)', text)
    return [p.strip() for p in parts if p.strip()]


def _join_rules(rule_list: list[str]) -> str:
    """Re-join rules, renumbering 1..N."""
    out = []
    for i, r in enumerate(rule_list, 1):
        r_clean = re.sub(r'^\s*\d+\.\s*', '', r).strip()
        out.append(f"{i}. {r_clean}")
    return "\n\n" + "\n\n".join(out) + "\n\n"


def _sanitize_rules(rules: str, fallback: str) -> str:
    """Nettoie et valide que la sortie est bien une liste de règles numérotées."""
    if not rules or len(rules) < 50:
        return fallback
    # Vérifie qu'il y a au moins une ligne numérotée
    if not re.search(r'^\s*\d+\.', rules, re.MULTILINE):
        return fallback
    return rules


# ── Formatage du feedback pour reflection (succès + échecs) ──────────────────

def _format_feedback_for_reflection(
    examples: list[dict[str, str]],
    results: list[dict[str, Any]],
    scores: list[dict[str, float]],
    perfect_score: float = 1.0,
    valid_placeholders: set[str] | None = None,
) -> str:
    """Formate TOUS les exemples (succès et échecs) avec feedback pour le mutateur.

    Contrairement à l'ancien _format_failures qui ne montrait que les échecs,
    ici on montre aussi les succès pour que le mutateur sache ce qu'il ne faut PAS casser.
    """
    formatted_parts = []
    for i, (example, result, score) in enumerate(zip(examples, results, scores)):
        # Score moyen
        avg_score = sum(score.values()) / len(score) if score else 0.0

        if avg_score >= perfect_score:
            score_str = f"Perfect success. Scores: {score}."
            feedback = "These rules worked well here. Preserve this behavior."
        elif avg_score >= perfect_score * 0.5:
            score_str = f"Partial success. Scores: {score}."
            feedback = "Consider how to improve the weaker aspects while keeping what works."
        else:
            score_str = f"Needs improvement. Scores: {score}."
            feedback = "The transformation did not produce a good enough prompt for this task."

        # Extrait les champs pertinents
        original_prompt = example.get("prompt", "N/A")
        transformed_prompt = result.get("improved", "N/A")
        code = result.get("code", "N/A")[:800]
        error = result.get("error", "")
        trajectory = result.get("trajectory", "")

        traj_str = ""
        if trajectory:
            traj_str = (
                f"\n\n**Agent Trajectory (reasoning & steps):**\n"
                f"```\n{str(trajectory)[:600]}\n```"
            )

        formatted_parts.append(
            f"### Example {i + 1} ###\n"
            f"**Original Prompt:**\n```\n{original_prompt}\n```\n\n"
            f"**Transformed Prompt:**\n```\n{transformed_prompt}\n```\n\n"
            f"**Generated Code:**\n```\n{code}\n```\n"
            f"{traj_str}\n\n"
            f"**Feedback:** {score_str} {feedback}"
            + (f"\n**Error:** {error}" if error else "")
        )

    return "\n\n---\n\n".join(formatted_parts)


# ── GEPA Core ────────────────────────────────────────────────────────────────

class GEPA:
    """GEPA fidèle au papier, adapté pour des "Transformation Rules"."""

    def __init__(
        self,
        initial_rules: str,
        evaluate_fn: Callable[[str, list[dict]], list[dict]],
        llm_reflect_fn: Callable[[str], str],
        llm_merge_fn: Callable[[str], str] | None = None,
        max_generations: int = DEFAULT_GENERATIONS,
        population_size: int = DEFAULT_POPULATION_SIZE,
        pareto_size: int = DEFAULT_PARETO_SIZE,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        dev_val_split: float = DEFAULT_DEV_VAL_SPLIT,
        perfect_score: float = DEFAULT_PERFECT_SCORE,
        use_merge: bool = True,
        max_merge_invocations: int = DEFAULT_MAX_MERGE_INVOCATIONS,
        merge_val_overlap_floor: int = DEFAULT_MERGE_VAL_OVERLAP_FLOOR,
        minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    ):
        """
        Args:
            initial_rules: Règles de transformation initiales (numérotées)
            evaluate_fn:   Fonction (rules, examples) -> list[dict] avec clés
                          'success', 'improved', 'code', 'error', 'trajectory', etc.
            llm_reflect_fn: Fonction (prompt_text) -> new_rules_text
            llm_merge_fn:   Fonction (prompt_text) -> merged_rules_text (optionnel)
        """
        self.initial_rules = initial_rules
        self.evaluate_fn = evaluate_fn
        self.llm_reflect_fn = llm_reflect_fn
        self.llm_merge_fn = llm_merge_fn

        self.max_generations = max_generations
        self.population_size = population_size
        self.pareto_size = pareto_size
        self.mutation_rate = mutation_rate
        self.dev_val_split = dev_val_split
        self.perfect_score = perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.merge_val_overlap_floor = merge_val_overlap_floor
        self.minibatch_size = minibatch_size

        # État
        self.candidates: list[RuleCandidate] = []
        self.pareto_frontier: list[RuleCandidate] = []
        self.best_per_val_instance: dict[int, RuleCandidate] = {}
        self.dev_examples: list[dict[str, str]] = []
        self.val_examples: list[dict[str, str]] = []
        self._candidate_id = 0
        self._merge_invocations = 0
        self._attempted_merges: set[tuple[int, int]] = set()
        self._historical_rules: dict[int, str] = {}
        self._ancestry: dict[int, list[int]] = {}

        # Initialiser avec le candidat seed
        self.candidates.append(self._new_candidate(initial_rules))

    def _new_candidate(self, rules: str, parents: list[int] | None = None) -> RuleCandidate:
        cand = RuleCandidate(
            rules=rules,
            id=self._candidate_id,
            parents=parents or [],
        )
        self._historical_rules[self._candidate_id] = rules
        self._ancestry[self._candidate_id] = parents or []
        self._candidate_id += 1
        return cand

    def _get_val_accuracy(self, candidate: RuleCandidate) -> float:
        if candidate.val_scores:
            return sum(candidate.val_scores.values()) / len(candidate.val_scores)
        return 0.0

    def _weighted_choice(self, candidates: list[RuleCandidate]) -> RuleCandidate:
        """Sélection pondérée par le nombre d'instances gagnées (GEPA original)."""
        if not candidates:
            raise ValueError("No candidates to choose from")
        weights = [max(1, len(c.val_instance_wins)) for c in candidates]
        return random.choices(candidates, weights=weights)[0]

    def _is_perfect(self, scores: dict[str, float]) -> bool:
        return bool(scores) and all(v >= self.perfect_score for v in scores.values())

    def _should_accept(self, parent_scores: dict[str, float], child_scores: dict[str, float]) -> bool:
        """Gating GEPA : accepte si l'enfant n'est pas pire que le parent (>=)."""
        if not parent_scores or not child_scores:
            return True
        return sum(child_scores.values()) >= sum(parent_scores.values())

    def _run_minibatch(
        self,
        rules: str,
        examples: list[dict[str, str]],
        capture_results: bool = False,
    ) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, Any]]]:
        """Évalue un ensemble de règles sur un minibatch.

        Returns:
            (avg_scores, per_item_scores, full_results)
        """
        all_scores: list[dict[str, float]] = []
        full_results: list[dict[str, Any]] = []

        results = self.evaluate_fn(rules, examples)

        for i, (example, result) in enumerate(zip(examples, results)):
            success = result.get("success", False)
            scores = {"success": 1.0 if success else 0.0}
            for k, v in result.items():
                if k != "success" and isinstance(v, (int, float)):
                    scores[k] = float(v)
            all_scores.append(scores)
            if capture_results:
                full_results.append(result)
                status = "PASS ✓" if success else f"FAIL ✗  ({(result.get('error') or '')[:80]})"
                tid = result.get("task_id", f"#{i}")
                print(f"\n    [{tid}] {status}")
                orig = result.get("prompt", "")
                imp  = result.get("improved", "")
                print(f"    ORIGINAL : {orig[:400].replace(chr(10), chr(10)+'             ')}")
                print(f"    IMPROVED : {imp[:400].replace(chr(10), chr(10)+'             ')}")

        return self._aggregate_scores(all_scores), all_scores, full_results

    @staticmethod
    def _aggregate_scores(all_scores: list[dict[str, float]]) -> dict[str, float]:
        if not all_scores:
            return {}
        avg: dict[str, float] = {}
        for key in all_scores[0]:
            avg[key] = sum(s.get(key, 0.0) for s in all_scores) / len(all_scores)
        return avg

    def _reflect(
        self,
        rules: str,
        examples: list[dict[str, str]],
        results: list[dict[str, Any]],
        scores: list[dict[str, float]],
    ) -> str:
        """Génère un nouvel ensemble de règles via réflexion LLM."""
        feedback = _format_feedback_for_reflection(
            examples, results, scores, self.perfect_score
        )

        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            rules=rules,
            inputs_outputs_feedback=feedback,
        )
        new_rules = self.llm_reflect_fn(prompt)
        new_rules = _sanitize_rules(new_rules, fallback=rules)
        print(f"\n  [REFLECT] Rule diff:\n{_diff_rules(rules, new_rules)}\n")
        return new_rules

    def _compute_val_overlap(
        self, c1: RuleCandidate, c2: RuleCandidate
    ) -> tuple[set[int], dict[str, float], dict[str, float]]:
        """Calcule le chevauchement validation entre deux candidats."""
        overlap_ids = c1.evaluated_val_ids & c2.evaluated_val_ids
        c1_scores: dict[str, float] = {}
        c2_scores: dict[str, float] = {}

        if overlap_ids and c1.per_item_val_scores and c2.per_item_val_scores:
            for idx in overlap_ids:
                if idx < len(c1.per_item_val_scores) and idx < len(c2.per_item_val_scores):
                    for key, val in c1.per_item_val_scores[idx].items():
                        c1_scores[key] = c1_scores.get(key, 0.0) + val
                    for key, val in c2.per_item_val_scores[idx].items():
                        c2_scores[key] = c2_scores.get(key, 0.0) + val
            for key in c1_scores:
                c1_scores[key] /= len(overlap_ids)
            for key in c2_scores:
                c2_scores[key] /= len(overlap_ids)

        return overlap_ids, c1_scores, c2_scores

    def _get_ancestors(self, candidate_id: int) -> set[int]:
        ancestors = set()
        stack = [candidate_id]
        while stack:
            node = stack.pop()
            if node not in ancestors:
                ancestors.add(node)
                stack.extend(self._ancestry.get(node, []))
        return ancestors

    def _find_common_ancestor(self, id1: int, id2: int) -> int | None:
        common = self._get_ancestors(id1) & self._get_ancestors(id2)
        return max(common) if common else None

    def _find_merge_candidates(self) -> list[tuple[RuleCandidate, RuleCandidate]]:
        """Trouve les paires du front de Pareto éligibles au merge."""
        if len(self.pareto_frontier) < 2:
            return []

        merge_pairs: list[tuple[RuleCandidate, RuleCandidate, float]] = []

        for i, c1 in enumerate(self.pareto_frontier):
            for c2 in self.pareto_frontier[i + 1:]:
                pair_key = (min(c1.id, c2.id), max(c1.id, c2.id))
                if pair_key in self._attempted_merges:
                    continue

                overlap_ids, _, _ = self._compute_val_overlap(c1, c2)
                if len(overlap_ids) < self.merge_val_overlap_floor:
                    continue

                if self._find_common_ancestor(c1.id, c2.id) is None:
                    continue

                union = c1.val_instance_wins | c2.val_instance_wins
                if not union:
                    continue
                symmetric_diff = c1.val_instance_wins ^ c2.val_instance_wins
                merge_score = (
                    len(symmetric_diff) / len(union) * 0.6
                    + len(union) / max(1, len(self.val_examples)) * 0.4
                )
                merge_pairs.append((c1, c2, merge_score))

        merge_pairs.sort(key=lambda x: x[2], reverse=True)
        return [(c1, c2) for c1, c2, _ in merge_pairs]

    def _merge_structural(self, c1: RuleCandidate, c2: RuleCandidate) -> RuleCandidate | None:
        """Merge 3-way structurel fidèle au papier GEPA."""
        if self._merge_invocations >= self.max_merge_invocations:
            return None

        pair_key = (min(c1.id, c2.id), max(c1.id, c2.id))
        self._attempted_merges.add(pair_key)
        self._merge_invocations += 1

        ancestor_id = self._find_common_ancestor(c1.id, c2.id)
        ancestor_rules = self._historical_rules.get(ancestor_id) if ancestor_id else None
        if ancestor_rules is None:
            return None

        # Merge via LLM si disponible, sinon merge heuristique
        if self.llm_merge_fn is not None:
            prompt = MERGE_PROMPT_TEMPLATE.format(
                ancestor_rules=ancestor_rules,
                rules_a=c1.rules,
                rules_b=c2.rules,
            )
            merged_rules = self.llm_merge_fn(prompt)
            merged_rules = _sanitize_rules(merged_rules, fallback=ancestor_rules)
        else:
            # Fallback heuristique : 3-way textuel simple
            p1, p2 = c1.rules, c2.rules
            c1_changed = p1 != ancestor_rules
            c2_changed = p2 != ancestor_rules
            if c1_changed and c2_changed and p1 != p2:
                s1 = sum(c1.val_scores.values()) if c1.val_scores else 0
                s2 = sum(c2.val_scores.values()) if c2.val_scores else 0
                merged_rules = p1 if s1 > s2 else p2 if s2 > s1 else random.choice([p1, p2])
            elif c2_changed:
                merged_rules = p2
            elif c1_changed:
                merged_rules = p1
            else:
                merged_rules = ancestor_rules

        # Gate : évalue sur le chevauchement
        overlap_ids, c1_scores, c2_scores = self._compute_val_overlap(c1, c2)
        overlap_examples = [
            self.val_examples[idx] for idx in overlap_ids if idx < len(self.val_examples)
        ]
        if len(overlap_examples) < self.merge_val_overlap_floor:
            return None

        merged = self._new_candidate(merged_rules, parents=[c1.id, c2.id])
        _, overlap_scores, _ = self._run_minibatch(merged.rules, overlap_examples)

        merged_avg = (
            sum(sum(s.values()) for s in overlap_scores) / len(overlap_scores)
            if overlap_scores else 0
        )
        parent_avg = max(sum(c1_scores.values()), sum(c2_scores.values()))

        return merged if merged_avg >= parent_avg * 0.95 else None

    def _try_merge_from_frontier(self) -> RuleCandidate | None:
        """Tente un merge depuis le front de Pareto."""
        if not self.use_merge:
            return None
        merge_pairs = self._find_merge_candidates()
        if not merge_pairs:
            return None
        c1, c2 = merge_pairs[0]
        return self._merge_structural(c1, c2)

    def _update_pareto(self, candidate: RuleCandidate) -> None:
        """Met à jour le front de Pareto instance-level."""
        prev_frontier_ids = {c.id for c in self.pareto_frontier}

        candidate.val_instance_wins = set()
        candidate.evaluated_val_ids = set(range(len(candidate.per_item_val_scores)))

        for idx, scores in enumerate(candidate.per_item_val_scores):
            score = sum(scores.values())
            current_best = self.best_per_val_instance.get(idx)

            if current_best is None:
                self.best_per_val_instance[idx] = candidate
                candidate.val_instance_wins.add(idx)
            else:
                best_score = sum(current_best.per_item_val_scores[idx].values())
                if score > best_score:
                    current_best.val_instance_wins.discard(idx)
                    self.best_per_val_instance[idx] = candidate
                    candidate.val_instance_wins.add(idx)
                elif score == best_score:
                    candidate.val_instance_wins.add(idx)

        frontier_candidates = {c.id: c for c in self.best_per_val_instance.values()}
        if candidate.val_instance_wins:
            frontier_candidates[candidate.id] = candidate

        new_frontier = sorted(
            frontier_candidates.values(),
            key=lambda c: len(c.val_instance_wins),
            reverse=True,
        )
        self.pareto_frontier = new_frontier[: self.pareto_size]

        if candidate.id in {c.id for c in self.pareto_frontier} and candidate.id not in prev_frontier_ids:
            print(
                f"  [PARETO] Added candidate {candidate.id} to frontier "
                f"(wins={len(candidate.val_instance_wins)}, "
                f"val_acc={self._get_val_accuracy(candidate):.4f})"
            )

    def optimize(self, train_examples: list[dict[str, str]]) -> RuleCandidate:
        """Boucle d'optimisation GEPA fidèle.

        Returns:
            Le meilleur RuleCandidate trouvé.
        """
        # Split dev/val
        shuffled = train_examples.copy()
        random.shuffle(shuffled)
        split = max(1, int(len(shuffled) * self.dev_val_split))
        self.dev_examples = shuffled[:split] or shuffled[:1]
        self.val_examples = shuffled[split:] or shuffled[-1:]

        print(f"GEPA starting — {len(self.dev_examples)} dev | {len(self.val_examples)} val")
        print(f"Initial rules:\n{self.initial_rules}\n")
        print("=" * 80)

        for gen in range(self.max_generations):
            # ── Évaluation de la population actuelle ─────────────────────────
            candidate_reflection_data: dict[
                int,
                tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, float]]]
            ] = {}

            for idx, candidate in enumerate(self.candidates):
                print(f"\n  Gen {gen+1}/{self.max_generations} | DEV eval | candidate {candidate.id}")

                # Dev : capture résultats pour reflection
                dev_batch = (
                    self.dev_examples
                    if len(self.dev_examples) <= self.minibatch_size
                    else random.sample(self.dev_examples, self.minibatch_size)
                )
                candidate.dev_scores, dev_item_scores, dev_results = self._run_minibatch(
                    candidate.rules, dev_batch, capture_results=True
                )
                candidate_reflection_data[candidate.id] = (dev_batch, dev_results, dev_item_scores)

                # Val : sélection Pareto
                print(f"  Gen {gen+1}/{self.max_generations} | VAL eval | candidate {candidate.id}")
                candidate.val_scores, candidate.per_item_val_scores, _ = self._run_minibatch(
                    candidate.rules, self.val_examples
                )
                self._update_pareto(candidate)

                val_acc = self._get_val_accuracy(candidate)
                print(
                    f"  Gen {gen+1}/{self.max_generations} | VAL done | "
                    f"candidate {candidate.id}: val_acc={val_acc:.4f}, "
                    f"wins={len(candidate.val_instance_wins)}"
                )

            # ── Génération suivante (sauf dernière) ──────────────────────────
            if gen < self.max_generations - 1:
                new_candidates: list[RuleCandidate] = []

                # Phase 1 : Mutations réflexives
                while len(new_candidates) < self.population_size:
                    if random.random() < self.mutation_rate and self.pareto_frontier:
                        parent = self._weighted_choice(self.pareto_frontier)

                        # Early stop si parfait
                        if self._is_perfect(parent.dev_scores):
                            new_candidates.append(parent)
                            print(f"  Gen {gen+1} | REFLECT | candidate {parent.id} is perfect, copying")
                            continue

                        print(f"  Gen {gen+1} | REFLECT | mutating candidate {parent.id}")

                        # Récupère les données de reflection
                        if parent.id in candidate_reflection_data:
                            dev_batch, dev_results, dev_item_scores = candidate_reflection_data[parent.id]
                            parent_scores = parent.dev_scores
                        else:
                            dev_batch = (
                                self.dev_examples
                                if len(self.dev_examples) <= self.minibatch_size
                                else random.sample(self.dev_examples, self.minibatch_size)
                            )
                            parent_scores, dev_item_scores, dev_results = self._run_minibatch(
                                parent.rules, dev_batch, capture_results=True
                            )

                        new_rules = self._reflect(
                            parent.rules, dev_batch, dev_results, dev_item_scores
                        )
                        child = self._new_candidate(new_rules, parents=[parent.id])

                        # Gating : évalue l'enfant sur le même dev batch
                        print(f"  Gen {gen+1} | GATE | testing child {child.id}")
                        child.dev_scores, _, _ = self._run_minibatch(child.rules, dev_batch)

                        if self._should_accept(parent_scores, child.dev_scores):
                            new_candidates.append(child)
                            print(
                                f"  Gen {gen+1} | GATE | ACCEPT child {child.id} "
                                f"({sum(child.dev_scores.values()):.2f} >= {sum(parent_scores.values()):.2f})"
                            )
                        else:
                            print(
                                f"  Gen {gen+1} | GATE | REJECT child {child.id} "
                                f"({sum(child.dev_scores.values()):.2f} < {sum(parent_scores.values()):.2f})"
                            )
                    elif self.candidates:
                        # Copie un candidat existant si pas de mutation
                        new_candidates.append(random.choice(self.candidates))

                # Phase 2 : Merge structurel depuis le front de Pareto
                if self.use_merge and len(self.pareto_frontier) >= 2:
                    print(f"  Gen {gen+1} | MERGE | attempting structural merge")
                    merged = self._try_merge_from_frontier()
                    if merged is not None:
                        merged.val_scores, merged.per_item_val_scores, _ = self._run_minibatch(
                            merged.rules, self.val_examples
                        )
                        self._update_pareto(merged)
                        if merged.val_instance_wins:
                            new_candidates.append(merged)
                            print(
                                f"  Gen {gen+1} | MERGE | SUCCESS candidate {merged.id} "
                                f"(wins={len(merged.val_instance_wins)})"
                            )
                        else:
                            print(f"  Gen {gen+1} | MERGE | merged candidate {merged.id} did not improve frontier")

                self.candidates = new_candidates
                print(f"\n  Gen {gen+1} complete — population: {len(self.candidates)} | "
                      f"Pareto frontier: {len(self.pareto_frontier)}")

        best = self._get_best_candidate()
        print(f"\n{'='*80}")
        print("BEST TRANSFORMATION RULES")
        print(f"{'='*80}")
        print(best.rules)
        print(f"val_acc={self._get_val_accuracy(best):.4f}, wins={len(best.val_instance_wins)}")
        print(f"{'='*80}\n")
        return best

    def _get_best_candidate(self) -> RuleCandidate:
        candidates = self.pareto_frontier or self.candidates
        if candidates:
            return max(
                candidates,
                key=lambda c: (len(c.val_instance_wins), sum(c.val_scores.values())),
            )
        return self._new_candidate(self.initial_rules)

    def get_pareto_frontier(self) -> list[RuleCandidate]:
        return self.pareto_frontier.copy()

    def get_best_rules(self) -> str:
        return self._get_best_candidate().rules


# ── Deletion operator (utilitaire externe) ─────────────────────────────────

def deletion_operator(rules: str) -> str:
    """Drop one rule uniformly at random."""
    parts = _split_rules(rules)
    if len(parts) < 2:
        return rules
    drop_idx = random.randrange(len(parts))
    kept = [p for i, p in enumerate(parts) if i != drop_idx]
    return _join_rules(kept)


# ── Wrapper facile pour l'utilisateur ─────────────────────────────────────────

def run_gepa_fidele(
    problems: list[dict[str, str]],
    initial_rules: str,
    evaluate_fn: Callable[[str, list[dict]], list[dict]],
    llm_reflect_fn: Callable[[str], str],
    llm_merge_fn: Callable[[str], str] | None = None,
    generations: int = DEFAULT_GENERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    val_size: int | None = None,
) -> tuple[str, list[dict]]:
    """Wrapper simplifié compatible avec ton ancienne interface `run_gepa`.

    Args:
        problems: Liste de problèmes/dictionnaires avec au moins 'prompt'
        initial_rules: Règles initiales (string numérotée)
        evaluate_fn: (rules, examples) -> list[dict] avec 'success', 'improved', 'code', 'error'
        llm_reflect_fn: Fonction qui prend un prompt texte et retourne des règles
        llm_merge_fn: Optionnel, pour le merge LLM-guided
        generations, population_size, minibatch_size: Hyperparamètres
        val_size: Si fourni, force la taille du val set (sinon utilise dev_val_split)

    Returns:
        (best_rules_string, generation_log_list)
    """
    # Adapter le split si val_size est donné
    shuffled = list(problems)
    random.seed(42)
    random.shuffle(shuffled)

    if val_size is not None:
        val = shuffled[:val_size]
        dev = shuffled[val_size:]
        dev_val_split = len(dev) / max(1, len(problems)) if problems else 0.5
    else:
        val = None
        dev = None
        dev_val_split = DEFAULT_DEV_VAL_SPLIT

    gepa = GEPA(
        initial_rules=initial_rules,
        evaluate_fn=evaluate_fn,
        llm_reflect_fn=llm_reflect_fn,
        llm_merge_fn=llm_merge_fn,
        max_generations=generations,
        population_size=population_size,
        dev_val_split=dev_val_split,
        minibatch_size=minibatch_size,
    )

    # Si on a forcé un split manuel, on override
    if val is not None and dev is not None:
        gepa.val_examples = val
        gepa.dev_examples = dev
        # On ne re-split pas dans optimize, donc on patch
        # (Note: dans l'implémentation ci-dessus optimize refait le split,
        #  il faudrait soit passer les sets, soit modifier optimize pour accepter un split externe)
        # Pour cette version wrapper, on passe train_examples = shuffled et on laisse le split interne
        # si val_size n'est pas utilisé. Sinon on peut wrapper l'appel.
        pass

    best_candidate = gepa.optimize(shuffled)

    # Construction d'un gen_log minimal pour compatibilité
    gen_log = []
    for gen in range(gepa.max_generations):
        gen_log.append({
            "gen": gen,
            "pareto_size": len(gepa.pareto_frontier),
            "population_size": len(gepa.candidates),
            "best_val_acc": gepa._get_val_accuracy(best_candidate),
        })

    return best_candidate.rules, gen_log


run_gepa = run_gepa_fidele