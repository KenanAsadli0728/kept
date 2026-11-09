"""Ledger notes. Rule: settle in value, never in words.

The LLM MAY phrase a note (KEPT_LLM=1 + key). Every candidate passes `validate_note`; apology / promise language,
or a missing/changed amount, is rejected and the deterministic template is used instead. Source is logged per entry,
so the rejection rate is a metric.
"""
import re

from .. import llm

FORBIDDEN = re.compile(
    r"\b(sorry|apolog\w*|regret\w*|we promise|promise[sd]?|we(?:'ll| will) (?:do|try) better|do better|improve|make it up|inconvenienc\w*|"
    r"unfortunate\w*|rest assured|we value|your patience|we guarantee|committed to|overpay\w*|wasting)\b", re.I)
MAX_LEN = 220


def validate_note(text: str, expected_amount_str=None) -> bool:
    if not text or len(text) > MAX_LEN:
        return False
    if FORBIDDEN.search(text):
        return False
    if expected_amount_str and expected_amount_str not in text:
        return False
    return True


SYSTEM = ("Rewrite the ledger note below in one short, neutral, factual sentence or two. Keep every number exactly. "
          "Do NOT apologise, do NOT promise anything, do NOT mention the future. Output the note only.")


def make_note(template: str, amount_str: str):
    """Returns (note, source) with source in {'template', 'llm', 'template_after_reject'}."""
    cand = llm.complete(template, system=SYSTEM, max_tokens=120)
    if cand is None:
        return template, "template"
    cand = cand.strip().strip('"')
    if validate_note(cand, amount_str):
        return cand, "llm"
    return template, "template_after_reject"
