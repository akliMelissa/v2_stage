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

from config import (
    GENERATIONS, POPULATION_SIZE, PARETO_SIZE, MINIBATCH_SIZE, VAL_SIZE,
    NUM_PROBLEMS, MUTATION_RATE, PERFECT_SCORE,
    MAX_MERGE_INVOCATIONS, MERGE_VAL_OVERLAP_FLOOR,
)
from prompts import MUTATE_RULES_PROMPT


def _diff_rules(old: str, new: str) -> str:
    # compare les deux versions de règles ligne par ligne et retourne les différences
    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="before", tofile="after", lineterm="",
    ))
    return "".join(diff) if diff else "(no change)"

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
    # nettoie la sortie du llm pour ne garder que la liste numérotée

    # si le llm a produit quelque chose de trop court ou vide, on garde les règles précédentes
    if not rules or len(rules) < 30:
        return fallback

    # on cherche le "1." qui marque le début de la liste — tout ce qui précède est du blabla llm
    m = re.search(r'(?m)^\s*1\.\s', rules)
    if not m:
        return fallback
    # on coupe tout ce qui est avant le premier "1."
    rules = rules[m.start():]
    # on retire le gras markdown (**texte** → texte)
    rules = re.sub(r'\*\*([^*]+)\*\*', r'\1', rules)
    # on retire les titres markdown (## Section → Section)
    rules = re.sub(r'(?m)^#+\s+', '', rules)
    rules = rules.strip()
    # dernier contrôle : si après nettoyage c'est encore trop court, fallback
    if len(rules) < 30:
        return fallback
    return rules


def _format_reflect_sections(
    results: list[dict],
    baseline_by_id: dict[str, bool],
) -> tuple[str, str, str]:
    # trie les résultats en 3 groupes pour donner du contexte au mutateur de règles
    helped, regressed, still_failing = [], [], []

    for r in results:
        tid = r.get("task_id", "")
        cur = r.get("success", False)
        # baseline_by_id contient le résultat sans transformation pour chaque problème
        base = baseline_by_id.get(tid)
        if base is None:
            # si on n'a pas de baseline pour ce problème, on ne peut pas classer → on ignore
            pass
        elif not base and cur:
            # baseline échoue, règles réussissent → les règles ont aidé
            helped.append(r)
        elif base and not cur:
            # baseline réussit, règles échouent → les règles ont cassé quelque chose
            regressed.append(r)
        elif not base and not cur:
            # les deux échouent → les règles n'ont pas encore résolu ce problème
            still_failing.append(r)
        # base=True et cur=True → stable_pass → aucun signal utile, on saute

    def _fmt_pass(items: list[dict]) -> str:
        # formate les cas qui sont passés de fail → pass (exemples positifs pour le mutateur)
        if not items:
            return "(none)"
        parts = []
        for r in items[:3]: # juste quelque exmples positifs 
            parts.append(
                f"Problem {r.get('task_id', '?')} (rules applied: {r.get('applied_rules', '?')}):\n"
                f"  Original : {r.get('prompt', '')}\n"
                f"  Improved : {r.get('improved', '')}\n"
                f"  Result   : PASSED"
            )
        return "\n\n".join(parts)

    def _fmt_fail(items: list[dict]) -> str:
        # formate les cas qui échouent avec le code généré, la solution correcte et l'erreur
        # la solution canonique permet au mutateur de voir ce que le prompt aurait dû produire
        if not items:
            return "(none)"
        parts = []
        for r in items:
            parts.append(
                f"Problem {r.get('task_id', '?')} (rules applied: {r.get('applied_rules', '?')}):\n"
                f"  Original          : {r.get('prompt', '')}\n"
                f"  Improved          : {r.get('improved', '')}\n"
                f"  Generated code    : {(r.get('code') or '')}\n"
                f"  Canonical solution: {(r.get('canonical_solution') or '(not available)')}\n"
                f"  Error             : {r.get('error', '')}"
            )
        return "\n\n".join(parts)

    return _fmt_pass(helped), _fmt_fail(regressed), _fmt_fail(still_failing)


# ── GEPA Core ────────────────────────────────────────────────────────────────

class GEPA:
    """GEPA fidèle au papier, adapté pour des "Transformation Rules"."""

    def __init__(
        self,
        initial_rules: str,
        evaluate_fn: Callable[[str, list[dict]], list[dict]],
        llm_reflect_fn: Callable[[str], str],
        llm_merge_fn: Callable[[str], str] | None = None,
        max_generations: int = GENERATIONS,
        population_size: int = POPULATION_SIZE,
        pareto_size: int = PARETO_SIZE,
        mutation_rate: float = MUTATION_RATE,
        dev_val_split: float = (NUM_PROBLEMS - VAL_SIZE) / max(1, NUM_PROBLEMS),
        perfect_score: float = PERFECT_SCORE,
        use_merge: bool = True,
        max_merge_invocations: int = MAX_MERGE_INVOCATIONS,
        merge_val_overlap_floor: int = MERGE_VAL_OVERLAP_FLOOR,
        minibatch_size: int = MINIBATCH_SIZE,
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

        # --- état interne ---
        # liste des candidats actifs dans la génération courante
        self.candidates: list[RuleCandidate] = []
        # ensemble des meilleurs candidats par instance de validation (front de pareto)
        self.pareto_frontier: list[RuleCandidate] = []
        # pour chaque index de problème val, quel candidat y est le meilleur
        self.best_per_val_instance: dict[int, RuleCandidate] = {}
        # problèmes utilisés pour la mutation/reflection (jamais vus par le pareto)
        self.dev_examples: list[dict[str, str]] = []
        # problèmes utilisés pour la sélection pareto (jamais vus par le mutateur)
        self.val_examples: list[dict[str, str]] = []
        # résultat baseline par task_id : True si le modèle réussissait sans transformation
        self.baseline_by_id: dict[str, bool] = {}
        # compteur pour donner un id unique à chaque candidat créé
        self._candidate_id = 0
        # nombre de merges déjà effectués (limité par max_merge_invocations)
        self._merge_invocations = 0
        # paires de candidats déjà mergées pour éviter de refaire le même merge
        self._attempted_merges: set[tuple[int, int]] = set()
        # historique des règles de chaque candidat (pour le merge 3-way)
        self._historical_rules: dict[int, str] = {}
        # arbre généalogique : qui sont les parents de chaque candidat
        self._ancestry: dict[int, list[int]] = {}

        # on démarre avec un seul candidat : les règles initiales
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
        # gate gepa : le candidat enfant est accepté seulement s'il fait au moins aussi bien que le parent
        # (>= et pas strictement > : une égalité suffit à accepter)
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
            # score binaire uniquement : 1.0 si le code passe tous les tests, 0.0 sinon
            # on n'utilise PAS Tests_Passed/n_Tests pour éviter de biaiser le pareto
            # (un problème avec 4/5 tests ne doit pas battre un problème avec 0/2 tests)
            success = result.get("success", False)
            all_scores.append({"success": 1.0 if success else 0.0})
            if capture_results:
                full_results.append(result)

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
        results: list[dict[str, Any]],
    ) -> str:
        # mutation des règles par le llm : on lui montre ce qui a aidé, ce qui a régressé, ce qui échoue encore
        # le llm lit les exemples et propose une version améliorée des règles

        # on trie les résultats en 3 catégories avec le contexte baseline
        helped, regressed, still_failing = _format_reflect_sections(
            results, self.baseline_by_id
        )
        passed = sum(1 for r in results if r.get("success"))

        # on construit le prompt de mutation avec toutes les catégories
        prompt = MUTATE_RULES_PROMPT.format(
            current_rules=rules,
            passed=passed,
            total=len(results),
            pass_improvements=helped,
            pass_regressions=regressed,
            fail_failures=still_failing,
        )
        new_rules = self.llm_reflect_fn(prompt)
        # on nettoie la sortie llm (retire le blabla avant le "1.", le markdown, etc.)
        new_rules = _sanitize_rules(new_rules, fallback=rules)
        # on affiche le diff pour voir ce que le llm a changé
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
            if merged_rules in (ancestor_rules, c1.rules, c2.rules):
                return None
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
        # met à jour le front de pareto instance par instance
        # chaque candidat "gagne" les instances de validation où il est le meilleur
        prev_frontier_ids = {c.id for c in self.pareto_frontier}

        candidate.val_instance_wins = set()
        # on note quels indices val ce candidat a été évalué sur
        candidate.evaluated_val_ids = set(range(len(candidate.per_item_val_scores)))

        for idx, scores in enumerate(candidate.per_item_val_scores):
            score = sum(scores.values())
            current_best = self.best_per_val_instance.get(idx)

            if current_best is None:
                # premier candidat à être évalué sur cette instance → il gagne par défaut
                self.best_per_val_instance[idx] = candidate
                candidate.val_instance_wins.add(idx)
            else:
                best_score = sum(current_best.per_item_val_scores[idx].values())
                if score > best_score:
                    # ce candidat bat le précédent meilleur → on transfère la victoire
                    current_best.val_instance_wins.discard(idx)
                    self.best_per_val_instance[idx] = candidate
                    candidate.val_instance_wins.add(idx)
                elif score == best_score:
                    # égalité : les deux gagnent cette instance
                    candidate.val_instance_wins.add(idx)

        # on reconstruit le front : tous les candidats qui ont au moins une victoire
        frontier_candidates = {c.id: c for c in self.best_per_val_instance.values()}
        if candidate.val_instance_wins:
            frontier_candidates[candidate.id] = candidate

        # on trie par nombre de victoires et on garde les meilleurs (pareto_size max)
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
        # on mélange les problèmes et on les coupe en deux : dev pour muter, val pour le pareto
        shuffled = train_examples.copy()
        random.shuffle(shuffled)
        split = max(1, int(len(shuffled) * self.dev_val_split))
        self.dev_examples = shuffled[:split] or shuffled[:1]
        self.val_examples = shuffled[split:] or shuffled[-1:]

        print(f"GEPA starting — {len(self.dev_examples)} dev | {len(self.val_examples)} val")
        print(f"Initial rules:\n{self.initial_rules}\n")
        print("=" * 80)

        for gen in range(self.max_generations):
            # on garde les données de reflection de chaque candidat pour les réutiliser
            # sans refaire un appel gpu si le même candidat est choisi comme parent
            candidate_reflection_data: dict[
                int,
                tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, float]]]
            ] = {}

            for idx, candidate in enumerate(self.candidates):
                print(f"\n  Gen {gen+1}/{self.max_generations} | DEV eval | candidate {candidate.id}")

                # évaluation sur un sous-ensemble du dev (minibatch) pour capturer les résultats de mutation
                dev_batch = (
                    self.dev_examples
                    if len(self.dev_examples) <= self.minibatch_size
                    else random.sample(self.dev_examples, self.minibatch_size)
                )
                candidate.dev_scores, dev_item_scores, dev_results = self._run_minibatch(
                    candidate.rules, dev_batch, capture_results=True
                )
                # on sauvegarde les données de dev pour éviter de refaire l'éval si ce candidat est choisi parent
                candidate_reflection_data[candidate.id] = (dev_batch, dev_results, dev_item_scores)

                # évaluation sur tout le val pour mettre à jour le front de pareto
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

            # on ne génère pas de nouvelle population après la dernière génération
            if gen < self.max_generations - 1:
                new_candidates: list[RuleCandidate] = []

                # phase 1 : on génère population_size nouveaux candidats par mutation réflexive
                while len(new_candidates) < self.population_size:
                    if random.random() < self.mutation_rate and self.pareto_frontier:
                        # on choisit un parent dans le front de pareto, pondéré par ses victoires
                        parent = self._weighted_choice(self.pareto_frontier)

                        # si le parent a un score parfait sur dev, pas la peine de muter
                        if self._is_perfect(parent.dev_scores):
                            new_candidates.append(parent)
                            print(f"  Gen {gen+1} | REFLECT | candidate {parent.id} is perfect, copying")
                            continue

                        print(f"  Gen {gen+1} | REFLECT | mutating candidate {parent.id}")

                        # on réutilise les données de dev déjà calculées si disponibles
                        if parent.id in candidate_reflection_data:
                            dev_batch, dev_results, dev_item_scores = candidate_reflection_data[parent.id]
                            parent_scores = parent.dev_scores
                        else:
                            # sinon on refait l'éval (ça ne devrait pas arriver souvent)
                            dev_batch = (
                                self.dev_examples
                                if len(self.dev_examples) <= self.minibatch_size
                                else random.sample(self.dev_examples, self.minibatch_size)
                            )
                            parent_scores, dev_item_scores, dev_results = self._run_minibatch(
                                parent.rules, dev_batch, capture_results=True
                            )

                        # le llm lit les résultats et propose une version améliorée des règles
                        new_rules = self._reflect(parent.rules, dev_results)
                        child = self._new_candidate(new_rules, parents=[parent.id])

                        # gate : on évalue l'enfant sur le même batch pour vérifier qu'il n'est pas pire
                        print(f"  Gen {gen+1} | GATE | testing child {child.id}")
                        child.dev_scores, _, _ = self._run_minibatch(child.rules, dev_batch)

                        if self._should_accept(parent_scores, child.dev_scores):
                            new_candidates.append(child)
                            print(
                                f"  Gen {gen+1} | GATE | ACCEPT child {child.id} "
                                f"({sum(child.dev_scores.values()):.2f} >= {sum(parent_scores.values()):.2f})"
                            )
                        else:
                            # l'enfant est pire → on l'abandonne (le parent reste dans le pareto)
                            print(
                                f"  Gen {gen+1} | GATE | REJECT child {child.id} "
                                f"({sum(child.dev_scores.values()):.2f} < {sum(parent_scores.values()):.2f})"
                            )
                    elif self.candidates:
                        # pas de mutation cette fois : on copie un candidat existant
                        new_candidates.append(random.choice(self.candidates))

                # phase 2 : on essaie de merger deux candidats du front de pareto
                # l'idée est de combiner les règles qui marchent sur des instances différentes
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
    generations: int = GENERATIONS,
    population_size: int = POPULATION_SIZE,
    minibatch_size: int = MINIBATCH_SIZE,
    val_size: int | None = None,
    baseline_results: list[dict] | None = None,
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
        baseline_results: Optionnel. Résultats baseline (sans transformation) avec task_id et
            success. Permet au mutateur de distinguer Pass→Fail (régressions) de Fail→Fail.

    Returns:
        (best_rules_string, generation_log_list)
    """
    # on mélange les problèmes avec une graine fixe pour la reproductibilité
    shuffled = list(problems)
    random.seed(42)
    random.shuffle(shuffled)

    # dev_val_split = proportion du total allouée au dev (optimize() l'utilise pour re-couper en interne)
    if val_size is not None:
        dev_val_split = (len(problems) - val_size) / max(1, len(problems))
    else:
        dev_val_split = (NUM_PROBLEMS - VAL_SIZE) / max(1, NUM_PROBLEMS)

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

    # on injecte le baseline pour que le mutateur puisse classer les résultats en 3 catégories
    if baseline_results:
        gepa.baseline_by_id = {r["task_id"]: bool(r["success"]) for r in baseline_results}

    best_candidate = gepa.optimize(shuffled)

    # on construit un log minimal par génération pour la sauvegarde dans gen_log.json
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