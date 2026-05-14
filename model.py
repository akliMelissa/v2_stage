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
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
)
model.eval()
print("Model ready.")


# ── Call helpers ──────────────────────────────────────────────────────────────

def call_llm(prompt: str, temperature: float = 0.3, max_new_tokens: int | None = None) -> str:
    """One generation call. Returns the model's text reply (no preamble)."""
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    try:
        messages = [{"role": "user", "content": prompt}]
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
              max_new_tokens: int | None = None) -> str:
    """call_llm with retries on empty output."""
    for _ in range(retries):
        out = call_llm(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        if out:
            return out
        time.sleep(1)
    return ""
