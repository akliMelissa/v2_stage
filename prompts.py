"""
prompts.py — Initial transformation rules and prompt templates.

These are kept VERBATIM from the original script. Do not change wording
unless you also intend to change GEPA behavior.
"""

# ── Initial transformation rules ──────────────────────────────────────────────
# They do NOT add new information, new examples, new constraints, or new types.

INITIAL_TRANSFORMATION_RULES = """

1. Rename vague variables in the prose to descriptive ones (s→sequence, n→count, x→value).
   Do NOT change the function or class name in the starter code.

2. Replace domain-specific nouns with generic algorithmic equivalents:
   boxes→containers, graph→network, coins→items, grid→2D matrix.

3. Soften precise algorithmic language so the model picks its own correct strategy
   instead of a broken memorized template:
   "minimum cost path"→"cheapest route",
   "bipartite"→"split into two groups".

4. Restate how the input arrives if the original is ambiguous about parsing:
   "Input is a Python list, already parsed. Iterate with indices starting at 0."
   Only rephrase what's already implied — do not invent new format details.

5. Rewrite vague constraints already in the problem as explicit boundaries:
   "divisors"→"integers strictly smaller than x that divide x evenly".
   Rewrite negatives as positives: "not greater than"→"less than or equal to".


"""

# ── Prompt for applying rules to a single problem ─────────────────────────────

APPLY_RULES_PROMPT = """You are a prompt engineering expert at improving competitive programming prompts.

TRANSFORMATION RULES:
{rules}

ORIGINAL PROMPT:
{original_prompt}

Apply only the relevant transformation rules above (at least more than one rule MUST be applied ) to improve the original prompt.
CRITICAL: Do NOT add new examples, new constraints, new test cases, or any information
not already present in the original.
Output ONLY the improved prompt. No explanation, no preamble."""

# ── Prompt for mutating the rules themselves ──────────────────────────────────

MUTATE_RULES_PROMPT = """You are optimizing transformation rules for competitive programming prompts.

CURRENT TRANSFORMATION RULES:
{current_rules}

These rules were applied to {batch_size} coding problems.
Results: {passed}/{batch_size} generated correct code after transformation.

FAILED PROBLEMS:
For each failure you will see:
- ORIGINAL PROMPT: the original problem before transformation
- IMPROVED PROMPT: what the rules transformed it into
- CODE GENERATED: what the model wrote from the improved prompt
- ERROR: why it failed

{failures}

Your task:
1. Compare ORIGINAL and IMPROVED PROMPT to see what the rules did.
2. Identify whether the transformation was harmful, insufficient, or unrelated to the failure.
3. Improve the rules to fix these specific failure patterns.

CRITICAL CONSTRAINTS (non-negotiable):
- Rules MUST apply to ANY problem, not just the failed ones shown.
- Do NOT mention specific problem types, keywords, or domains from the failures above.
- Do NOT add "specify the type of", "provide examples", or "list the steps" — these add new info.
- Do NOT suggest breaking problems into sub-steps or mentioning algorithms by name.

Guidelines:
- Add a new rule only if it's a general transformation that would help multiple problem types.
- Improve or remove rules that seem harmful.
- Rules must only reframe/clarify, never add new information, types, or solution steps.

Output ONLY the improved transformation rules. No explanation."""