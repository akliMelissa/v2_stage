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

2. Restate how the input arrives if the original is ambiguous about parsing:
   "Input is a Python list, already parsed. Iterate with indices starting at 0."
   Only rephrase what's already implied — do not invent new format details.

3. Rewrite vague constraints already in the problem as explicit boundaries:
   "divisors"→"integers strictly smaller than x that divide x evenly".
   Rewrite negatives as positives: "not greater than"→"less than or equal to".
"""

# ── Prompt for applying rules to a single problem ─────────────────────────────

APPLY_RULES_PROMPT = """You are a prompt engineering expert at improving competitive programming prompts.

TRANSFORMATION RULES:
{rules}

ORIGINAL PROMPT:
{original_prompt}

You MUST apply at least 1 rule if applicable. For each rule you apply, ask yourself: does this change make the problem CLEARER?
- If renaming a variable or term would make it LESS precise in context → apply it only if the new name is strictly more descriptive.
- If softening algorithmic language would cause ambiguity → apply it carefully, preserving the core meaning.
Apply each rule independently. Rules that would actively confuse the reader should be skipped for this problem. If no rule is applicable without harming clarity, return the prompt unchanged.
CRITICAL: Do NOT add new examples, new constraints, new test cases, or any information not already present in the original.
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
- REFERENCE SOLUTION: the correct solution, shown ONLY to help you reason about
  WHY the generated code is wrong. This is privileged information.

{failures}

ABSOLUTE PROHIBITIONS — VIOLATING ANY OF THESE INVALIDATES YOUR OUTPUT:
- NEVER reproduce, paraphrase, or hint at the REFERENCE SOLUTION in your rules.
- NEVER mention specific algorithms, data structures, function names, variable names,
  or code fragments from the reference solutions.
- NEVER include code blocks, def, class, return, for, while, or if statements in your rules.
- NEVER name specific problem types, domains, or keywords from the failures.
- Rules MUST be GENERAL transformations applicable to ANY problem, not hints about
  these specific problems.
- The reference solution is for YOUR INTERNAL REASONING ONLY. Use it to judge whether
  a rule made the prompt misleading. Then produce general rules without any leakage.

Your task — reason step by step for EACH failure (silently, do not output the reasoning):
1. Compare ORIGINAL PROMPT vs IMPROVED PROMPT word by word. Identify exactly what changed
   (renamed terms, softened language, replaced nouns, etc.) and which rule NUMBER caused it.
2. Compare CODE GENERATED to the REFERENCE SOLUTION. Where did the generated code diverge
   from correctness? Did a rule cause that divergence by making a precise term vague?
   - If YES → that rule is HARMFUL. Remove it or restrict its scope.
   - If the prompt barely changed → the rule was irrelevant; the failure has another cause.
   - If a rename made a precise term vague → harmful.
3. A rule that causes confusion or wrong code across multiple failures MUST be REMOVED or
   its wording tightened so it only applies when it genuinely helps.
4. Only add a new rule if it would fix a pattern visible in multiple failures, expressed
   as a GENERAL transformation.

CRITICAL CONSTRAINTS (non-negotiable):
- Rules MUST apply to ANY problem, not just the failed ones shown.
- Do NOT mention specific problem types, keywords, or domains from the failures above.
- Do NOT add "specify the type of", "provide examples", or "list the steps" — these add new info.
- Do NOT suggest breaking problems into sub-steps or mentioning algorithms by name.

Guidelines:
- Add a new rule only if it's a general transformation that would help multiple problem types.
- Improve or remove rules that seem harmful.
- Rules must only reframe/clarify, never add new information, types, or solution steps.

Output ONLY the improved transformation rules as numbered items. No explanation, no code,
no references to specific problems, no mention of the reference solutions."""