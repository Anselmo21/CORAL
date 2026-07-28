"""Tests for check_generation_accuracy.score_graph and helpers.

The LLM call inside score_graph is stubbed via monkeypatch — no real inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import check_generation_accuracy as cga
from graph_ops import Graph


# ------------------------------------------------------------------ #
# Fixtures                                                           #
# ------------------------------------------------------------------ #

PLAUSIBLE = ["Ans-A", "Ans-B", "Ans-C"]
QUESTION = "What is the best X?"
SOURCES = "Source 1: sample text."


def _graph_from(nodes, edges) -> Graph:
    return Graph(
        nodes_by_id={n["id"]: n for n in nodes},
        edges=list(edges),
        meta={"graph": {"nodes": nodes, "edges": edges}},
    )


@pytest.fixture
def three_answer_graph() -> Graph:
    """Graph where each answer has a unique supporter set:
      a1 <- c1, c2
      a2 <- c3
      a3 <- c4, c5
    """
    nodes = [
        {"id": "q",   "layer": "1", "type": "question", "label": QUESTION},
        {"id": "sq1", "layer": "2", "type": "subq", "label": "sq1"},
        {"id": "c1",  "layer": "3", "type": "criterion", "label": "cheap"},
        {"id": "c2",  "layer": "3", "type": "criterion", "label": "fast"},
        {"id": "c3",  "layer": "3", "type": "criterion", "label": "safe"},
        {"id": "c4",  "layer": "3", "type": "criterion", "label": "custom"},
        {"id": "c5",  "layer": "3", "type": "criterion", "label": "novel"},
        {"id": "a1",  "layer": "4", "type": "answer", "label": "Ans-A"},
        {"id": "a2",  "layer": "4", "type": "answer", "label": "Ans-B"},
        {"id": "a3",  "layer": "4", "type": "answer", "label": "Ans-C"},
    ]
    edges = [
        {"source": "q", "target": "sq1"},
        {"source": "sq1", "target": "c1"}, {"source": "sq1", "target": "c2"},
        {"source": "sq1", "target": "c3"}, {"source": "sq1", "target": "c4"},
        {"source": "sq1", "target": "c5"},
        {"source": "c1", "target": "a1"}, {"source": "c2", "target": "a1"},
        {"source": "c3", "target": "a2"},
        {"source": "c4", "target": "a3"}, {"source": "c5", "target": "a3"},
    ]
    return _graph_from(nodes, edges)


def _score(graph, monkeypatch, raw_picker):
    """Helper: adapt an old-style picker (uses kwargs) to the new
    (supporters, plausible_answers) signature and call score_graph."""
    def _adapter(supporters, plausible_answers):
        return raw_picker(atoms=supporters, plausible_answers=plausible_answers)
    return cga.score_graph(picker=_adapter, graph=graph, plausible_answers=PLAUSIBLE)


# ------------------------------------------------------------------ #
# Tests                                                              #
# ------------------------------------------------------------------ #

def test_perfect_score_all_answers_correctly_supported(three_answer_graph, monkeypatch):
    """When the ranking prompt always returns the label matching the tested
    answer, we should score 3/3."""
    call_log = []

    def picker(*, atoms, plausible_answers, **_kw):
        call_log.append(tuple(sorted(atoms)))
        # Route by supporter fingerprint back to the intended answer
        atomset = set(atoms)
        if atomset == {"cheap", "fast"}:
            return 0, plausible_answers[0]     # a1 -> Ans-A
        if atomset == {"safe"}:
            return 1, plausible_answers[1]     # a2 -> Ans-B
        if atomset == {"custom", "novel"}:
            return 2, plausible_answers[2]     # a3 -> Ans-C
        raise AssertionError(f"unexpected atoms: {atoms}")

    matches, total, records = _score(three_answer_graph, monkeypatch, picker)
    assert (matches, total) == (3, 3)
    assert all(r["matched"] for r in records)
    assert len(call_log) == 3
    # Each record carries the fields we depend on downstream
    r0 = records[0]
    assert r0["answer_id"] == "a1"
    assert r0["supporters"] == ["cheap", "fast"]
    assert r0["picked_label"] == "Ans-A"
    assert r0["picked_index"] == 0
    assert r0["error"] is None


def test_partial_miss_wrong_top1(three_answer_graph, monkeypatch):
    """LLM picks the wrong answer for a2 → 2/3."""
    def picker(*, atoms, plausible_answers, **_kw):
        atomset = set(atoms)
        if atomset == {"cheap", "fast"}:
            return 0, plausible_answers[0]     # match
        if atomset == {"safe"}:
            return 0, plausible_answers[0]     # miss: expected Ans-B
        return 2, plausible_answers[2]         # match

    matches, total, records = _score(three_answer_graph, monkeypatch, picker)
    assert (matches, total) == (2, 3)
    missed = [r for r in records if not r["matched"]]
    assert len(missed) == 1
    assert missed[0]["answer_id"] == "a2"
    assert missed[0]["picked_label"] == "Ans-A"


def test_answer_label_not_in_plausible_answers(monkeypatch):
    """Answer node whose label isn't in plausible_answers is marked as an error
    record and the LLM is NOT called for it."""
    nodes = [
        {"id": "q",  "layer": "1", "type": "question", "label": QUESTION},
        {"id": "sq1","layer": "2", "type": "subq", "label": "sq1"},
        {"id": "c1", "layer": "3", "type": "criterion", "label": "x"},
        {"id": "c2", "layer": "3", "type": "criterion", "label": "y"},
        # a1 matches plausible; a2 is a "rogue" invented answer
        {"id": "a1", "layer": "4", "type": "answer", "label": "Ans-A"},
        {"id": "a2", "layer": "4", "type": "answer", "label": "Rogue"},
    ]
    edges = [
        {"source": "q", "target": "sq1"},
        {"source": "sq1", "target": "c1"}, {"source": "sq1", "target": "c2"},
        {"source": "c1", "target": "a1"}, {"source": "c2", "target": "a1"},
        {"source": "c1", "target": "a2"}, {"source": "c2", "target": "a2"},
    ]
    graph = _graph_from(nodes, edges)

    calls = []
    def picker(*, atoms, plausible_answers, **_kw):
        calls.append(True)
        return 0, plausible_answers[0]

    matches, total, records = _score(graph, monkeypatch, picker)
    assert (matches, total) == (1, 2)
    assert len(calls) == 1  # only a1 was tested; a2 shortcircuited

    rogue = next(r for r in records if r["answer_id"] == "a2")
    assert rogue["matched"] is False
    assert rogue["error"] == "answer_not_in_plausible"
    assert rogue["picked"] is None


def test_llm_exception_is_captured_as_error(three_answer_graph, monkeypatch):
    """If pick_answer_with_all_atoms raises, that answer is scored as miss with
    error text; other answers are still tested."""
    def picker(*, atoms, plausible_answers, **_kw):
        if "safe" in atoms:
            raise RuntimeError("boom")
        return 0, plausible_answers[0] if "cheap" in atoms else plausible_answers[2]

    matches, total, records = _score(three_answer_graph, monkeypatch, picker)
    # a1 hits (cheap/fast → Ans-A), a2 raises, a3 hits (custom/novel → Ans-C)
    assert (matches, total) == (2, 3)
    err_rec = next(r for r in records if r["answer_id"] == "a2")
    assert err_rec["matched"] is False
    assert err_rec["error"] and err_rec["error"].startswith("llm_or_parse:")
    assert "boom" in err_rec["error"]


def test_answer_with_zero_supporters_passes_empty_atoms_list(monkeypatch):
    """An answer with no c->a edges gets ranked with atoms=[]. That's the
    'unsupported answer' scenario — we still call the LLM with empty criteria."""
    nodes = [
        {"id": "q",  "layer": "1", "type": "question", "label": QUESTION},
        {"id": "sq1","layer": "2", "type": "subq", "label": "sq1"},
        {"id": "c1", "layer": "3", "type": "criterion", "label": "x"},
        {"id": "a1", "layer": "4", "type": "answer", "label": "Ans-A"},
        {"id": "a2", "layer": "4", "type": "answer", "label": "Ans-B"},
    ]
    edges = [
        {"source": "q", "target": "sq1"},
        {"source": "sq1", "target": "c1"},
        {"source": "c1", "target": "a1"},
        # a2 intentionally has no c->a edges
    ]
    graph = _graph_from(nodes, edges)

    seen_atoms = []
    def picker(*, atoms, plausible_answers, **_kw):
        seen_atoms.append(list(atoms))
        # Both calls return Ans-A (LLM's default preference with empty criteria
        # collapses to the same answer, per the "not-random" caveat).
        return 0, plausible_answers[0]

    matches, total, records = _score(graph, monkeypatch, picker)
    assert (matches, total) == (1, 2)  # a1 matches, a2 doesn't
    # a2's atoms list was empty
    a2_rec = next(r for r in records if r["answer_id"] == "a2")
    assert a2_rec["supporters"] == []
    assert seen_atoms[1] == []  # confirm empty list actually reached picker


def test_supporters_are_deduplicated_and_sorted(three_answer_graph, monkeypatch):
    """atoms_for_answer returns a set; score_graph sorts it. Ensure records
    reflect a sorted deduped list."""
    calls = []
    def picker(*, atoms, plausible_answers, **_kw):
        calls.append(list(atoms))
        return 0, plausible_answers[0]
    _score(three_answer_graph, monkeypatch, picker)
    for atoms in calls:
        assert atoms == sorted(set(atoms))


# ------------------------------------------------------------------ #
# _fmt_pct                                                           #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("n,d,expected", [
    (0, 10, "0.0%"),
    (5, 10, "50.0%"),
    (1, 3, "33.3%"),
    (10, 10, "100.0%"),
    (0, 0, "N/A"),
    (3, 0, "N/A"),
])
def test_fmt_pct(n, d, expected):
    assert cga._fmt_pct(n, d) == expected
