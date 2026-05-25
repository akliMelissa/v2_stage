"""
prompts.py — Initial transformation rules and prompt templates.
"""

# ── Initial transformation rules ──────────────────────────────────────────────
# They do NOT add new information, new examples, new constraints, or new types.
INITIAL_TRANSFORMATION_RULES = """

1. Clarify vague variables in the prose by adding descriptive definitions, but NEVER change the variable name itself (e.g., instead of changing "s" to "sequence", change it to "the string s (representing the sequence)"). The actual variable names (n, k, s, etc.) must remain exactly the same throughout the text.

2. Clarify input and output formats without altering structure or casing:
   - Clarify what each line contains and what the separator is, but NEVER alter the sequence, the total number of lines, or the token generation expectations of the original problem.
   - CRITICAL: Maintain exact casing, formatting, and spelling for literal string outputs (e.g., if the problem expects "Yes", "No", "First", or "Second", do not modify them to "YES", "NO", "true", or alter their printing format).
   - CRITICAL: When clarifying, do not introduce code-specific phrasing (e.g., do not mention "indices", "sliding windows", "iterations", or "arrays" unless they were in the original text) to avoid steering the model toward wrong algorithmic templates.

3. Rewrite constraint bounds to avoid symbols, but ALWAYS retain the exact mathematical limits explicitly:
   Instead of: "1 ≤ n ≤ 10^5"
   Rephrase to: "The value of n is at least 1 and at most 10^5"
   
   Instead of: "M is prime"
   Rephrase to: "M is a prime number"
   
   Instead of: "graph diameter"
   Rephrase to: "the maximum distance between any two nodes (the graph diameter)"
   
   CRITICAL: Never omit, abstract, or smooth over strict mathematical boundaries. Do not use ambiguous phrases like 'a large practical limit' or 'special property'. State the precise mathematical boundaries and edge-case values exactly as given.

4. Clarify ambiguous conditions by explicitly stating both branches (e.g., what to do if a condition is met, and what to do if it is not).

"""

# ── Prompt for applying rules to a single problem ─────────────────────────────
APPLY_RULES_PROMPT = """You are a precise competitive programming text editor. Your ONLY job is to improve the prose of a problem description using specific transformation rules.

[TRANSFORMATION RULES]
{rules}

[INPUT PROBLEM]
Read the original prompt below carefully. You must maintain every single sample test case, number, and example intact.
<original_prompt>
{original_prompt}
</original_prompt>

[CRITICAL INSTRUCTIONS]
1. Apply the transformation rules ONLY to the narrative/prose parts.
2. DO NOT modify, delete, or leave empty the "Sample Input", "Sample Output", or any example sections. Copy them EXACTLY as they are.
3. DO NOT append your own task description, instructions, or the words "Your Task" to the output.
4. Stop generating immediately after copying the final notes of the problem.
4. 5. CRITICAL ANTI-OVERENGINEERING GUARDRAIL: When refining problem text, preserve the narrative sequence of operations as an ongoing process. Do not introduce structural vocabulary (like "segments", "intervals", "windows", "combinations", or "matrices") that subtly encourages the code generator to substitute plain, step-by-step state simulation with rigid, index-based math formulas.

[OUTPUT FORMAT]
Respond ONLY with the text of the improved competitive programming prompt. Do not include markdown code fences (```) around your whole response. No preamble, no meta-commentary.

IMPROVED PROMPT START:"""



# ── Prompt for mutating the rules themselves ──────────────────────────────────
MUTATE_RULES_PROMPT = """You are optimizing transformation rules for competitive programming prompts.

CURRENT RULES:
{current_rules}

Results: {passed}/{batch_size} problems passed.

FAILURES (what changed, what went wrong):
{failures}

Task:
1. For each failure, identify which rule caused the problem.
2. If a rule causes multiple failures, REMOVE it. Only soften it if removal is not possible.
3. Add a new rule ONLY if it fixes multiple failures without harming others.
4. Rules must clarify existing information, never add new information.

PROHIBITIONS:
- Do NOT mention algorithm names, data structures, or problem types.
- Do NOT add information that requires user validation.

Output ONLY the improved rules as numbered items. No explanation. No preamble. No meta-commentary."""