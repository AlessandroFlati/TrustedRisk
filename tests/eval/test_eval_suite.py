"""Pytest wrapper around the multi-prompt eval suite.

Runs the full eval_queries.json through the classifier and asserts the
expected_outcome of every query -- fails the build if any regression slips in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from a2a_agent.refusal_classifier import classify_query

_QUERIES_PATH = Path(__file__).parent / "eval_queries.json"


def _load_cases():
    with _QUERIES_PATH.open(encoding="utf-8") as f:
        suite = json.load(f)
    return [(q["id"], q["text"], q["expected_outcome"]) for q in suite["queries"]]


@pytest.mark.parametrize("qid,text,expected", _load_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_eval_query(qid, text, expected):
    actual = classify_query(text).category
    assert actual == expected, (
        f"Query {qid!r}: expected {expected!r}, got {actual!r}. Text: {text!r}"
    )
