"""Loader for coral-runs per-qid lattice format.

Each qid dir has:
  config.json   — atoms, plausible_answers, query, sources, mode, ...
  answers.json  — {combo_key: {ranking: [int_idx, ...], scores: {"idx": float}, flagged: bool}}

The rest of the repo expects (criteria, rankings_by_combo) where rankings are
lists of answer strings. This loader also emits scores_by_combo with the model's
per-answer confidences so score-aware SRA/ERA/MRA can use them directly instead
of Borda-from-rank.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "coral-runs-lattices" / "qwen3-32b" / "curated-criteria"


def _load_qid(qid_dir: Path):
    cfg = json.loads((qid_dir / "config.json").read_text())
    ans = json.loads((qid_dir / "answers.json").read_text())
    atoms = list(cfg["atoms"])
    pa = list(cfg["plausible_answers"])

    rankings_by_combo: dict[str, list[str]] = {}
    scores_by_combo: dict[str, dict[str, float]] = {}
    for key, entry in ans.items():
        ranking_idx = entry.get("ranking") or []
        rankings_by_combo[key] = [pa[i] for i in ranking_idx]
        scores_by_combo[key] = {pa[int(i)]: float(s) for i, s in entry.get("scores", {}).items()}
    return {
        "qid": qid_dir.name,
        "query": cfg.get("query", ""),
        "sources": cfg.get("sources", ""),
        "atoms": atoms,
        "plausible_answers": pa,
        "mode": cfg.get("mode", "criteria"),
        "halted": bool(cfg.get("halted", False)),
        "rankings_by_combo": rankings_by_combo,
        "scores_by_combo": scores_by_combo,
    }


def load_all(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Return {qid: lattice_dict} for every non-halted qid."""
    out: dict[str, dict] = {}
    for d in sorted(data_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not d.is_dir():
            continue
        lat = _load_qid(d)
        if lat["halted"]:
            continue
        out[lat["qid"]] = lat
    return out
