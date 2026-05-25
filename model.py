"""
model.py — Load the LLM once at import time and expose call helpers.

To swap models, change MODEL_NAME in config.py.
"""

import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, MAX_NEW_TOKENS, BATCH_SIZE

# ── Load model and tokenizer ──────────────────────────────────────────────────

print(f"Loading {MODEL_NAME} on GPU 0 and 1 ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
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
             system_prompt: str | None = None) -> str:
    """Single generation call."""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print("LLM error:", e)
        return ""


def _run_batch_chunk(texts: list[str], temperature: float) -> list[str]:
    """Run one chunk of prompts through the model."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for out_ids, in_ids in zip(out, inputs["input_ids"]):
        new_tokens = out_ids[len(in_ids):]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


def batch_call_llm(prompts: list[str], temperature: float = 0.3,
                   system_prompt: str | None = None) -> list[str]:
    """Batch generation — splits into chunks of BATCH_SIZE to avoid OOM."""
    if not prompts:
        return []
    texts = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
    results = []
    n_chunks = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        print(f"  [GPU] batch {i//BATCH_SIZE + 1}/{n_chunks} ({len(chunk)} prompts)...", flush=True)
        try:
            results.extend(_run_batch_chunk(chunk, temperature))
        except Exception as e:
            print(f"LLM batch error on chunk {i//BATCH_SIZE}: {e}")
            results.extend([""] * len(chunk))
    return results


def safe_call(prompt: str, retries: int = 2, temperature: float = 0.3,
              system_prompt: str | None = None) -> str:
    """call_llm with retries on empty output."""
    for _ in range(retries):
        out = call_llm(prompt, temperature=temperature,
                       system_prompt=system_prompt)
        if out:
            return out
        time.sleep(1)
    return ""
