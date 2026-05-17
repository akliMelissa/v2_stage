"""
model.py — Load the LLM once at import time and expose call helpers.

To swap models, change MODEL_NAME in config.py.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, DEVICE, MAX_NEW_TOKENS

# ── Load model and tokenizer ──────────────────────────────────────────────────

print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map={"": DEVICE},
)
model.eval()
print("Model ready.")


# ── System prompt for code generation ─────────────────────────────────────────

CODE_GEN_SYSTEM_PROMPT = (
    "You are an expert Python competitive programming assistant. "
    "When asked to write code, you ALWAYS include explicit imports at the top "
    "of your solution. Required imports whenever you use the corresponding name: "
    "`from functools import lru_cache, cache, reduce, partial`, "
    "`from collections import defaultdict, Counter, deque, OrderedDict`, "
    "`import heapq, bisect, math, re, sys, itertools`, "
    "`from typing import List, Dict, Tuple, Set, Optional`. "
    "Never use a decorator like @lru_cache or a name like defaultdict, heapq, "
    "Counter, deque without importing it first. "
    "Wrap your final solution in a ```python ... ``` code block."
)


# ── Call helpers ──────────────────────────────────────────────────────────────

def call_llm(prompt: str, temperature: float = 0.3,
             max_new_tokens: int | None = None,
             system_prompt: str | None = None) -> str:
    """One generation call. Returns the model's text reply (no preamble)."""
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print("LLM error:", e)
        return ""


def safe_call(prompt: str, retries: int = 2, temperature: float = 0.3,
              max_new_tokens: int | None = None,
              system_prompt: str | None = None) -> str:
    """call_llm with retries on empty output."""
    for _ in range(retries):
        out = call_llm(prompt, temperature=temperature,
                       max_new_tokens=max_new_tokens,
                       system_prompt=system_prompt)
        if out:
            return out
        time.sleep(1)
    return ""