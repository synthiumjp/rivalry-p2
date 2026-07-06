#!/usr/bin/env python3
"""
answer_span.py

Single source of truth for where a generated answer ends. The pipeline's
recurring failure mode is generation-behaviour artifacts (degeneration into a
fabricated second turn) contaminating measurements. This rule cuts every
generation at the answer span BEFORE scoring or position-labelling.

Answer span = generated text up to the FIRST of: a newline, a chat/role marker
("Human:", "Assistant:", etc.), or EOS. Everything after is discarded.

Two entry points, both using the same markers:
  truncate_answer_text(text)                 -> str  (string level, for scoring)
  answer_span_end_index(gen_ids, tok, text)  -> int  (token level, for capture)
"""

STOP_MARKERS = [
    "\n",
    "Human:", "Assistant:", "human:", "assistant:",
    "HUMAN:", "ASSISTANT:",
    "<|", "###",
]

def truncate_answer_text(text: str) -> str:
    """Cut decoded completion at the first stop marker. Case-insensitive."""
    if not text:
        return ""
    low = text.lower()
    cut = len(text)
    for m in STOP_MARKERS:
        i = low.find(m.lower())
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()

def answer_span_end_index(gen_ids, tokenizer, answer_text: str) -> int:
    """Smallest generated-token index whose cumulative decode covers the answer
    span. That token is the last answer token (the correct last_answer_tok).
    Returns 0 if the answer is empty (degenerate)."""
    if not answer_text:
        return 0
    target = len(answer_text)
    for k in range(len(gen_ids)):
        partial = tokenizer.decode(gen_ids[:k + 1], skip_special_tokens=True).strip()
        if len(partial) >= target:
            return k
    return len(gen_ids) - 1
