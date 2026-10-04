from __future__ import annotations

from typing import Set

from experiments.counterfactual_eval.algorithms import sra_search, bfs_search, pyflagr_search
from experiments.counterfactual_eval.loader import lattice_from_run


def test() -> Set[str]:
    """Smoke test: load a toy lattice-like run and run BFS."""
    run = {
        "criteria_used": ["effectiveness", "price", "side effects"],
        "rankings_by_combo": {
            "": ["medications", "exercise", "therapy"],
            "effectiveness": ["medications", "exercise", "therapy"],
            "price": ["medications", "exercise", "therapy"],
            "side effects": ["exercise", "therapy", "medications"],
            "effectiveness, price": ["exercise", "therapy", "medications"],
            "effectiveness, side effects": ["exercise", "therapy", "medications"],
            "price, side effects": ["exercise", "therapy", "medications"],
            "effectiveness, price, side effects": [
                "exercise",
                "therapy",
                "medications",
            ],
        },
    }
    lattice = lattice_from_run(
        run,
        source_path="tests/test_methods.py",
        lattice_id="test_lattice",
    )
    result = sra_search(lattice)
    return result.predicted_keys

def test_pyflagr() -> Set[str]:
    """Smoke test: load a toy lattice-like run and run PyFlagr-backed Borda."""

    # lattice size 4, so 2^4 = 16 nodes
    run = {
        "criteria_used": ["effectiveness", "price", "side effects", "safety"],
        "rankings_by_combo": {
            "": ["medications", "exercise", "therapy"],
            "effectiveness": ["medications", "exercise", "therapy"],
            "price": ["medications", "exercise", "therapy"],
            "side effects": ["exercise", "therapy", "medications"],
            "safety": ["exercise", "therapy", "medications"],
            "effectiveness, price": ["exercise", "therapy", "medications"],
            "effectiveness, side effects": ["exercise", "therapy", "medications"],
            "price, side effects": ["exercise", "therapy", "medications"],
            "price, safety": ["exercise", "therapy", "medications"],
            "side effects, safety": ["exercise", "therapy", "medications"],
            "effectiveness, price, side effects, safety": [
                "exercise",
                "therapy",
                "medications",
            ],
            "effectiveness, price, side effects": ["exercise", "therapy", "medications"],
            "effectiveness, price, safety": ["exercise", "therapy", "medications"],
            "effectiveness, side effects, safety": ["exercise", "therapy", "medications"],
            "price, side effects, safety": ["exercise", "therapy", "medications"],
            "effectiveness, price, side effects, safety": ["exercise", "therapy", "medications"],
        },
    }
    lattice = lattice_from_run(
        run,
        source_path="tests/test_methods.py",
        lattice_id="test_lattice_pyflagr",
    )
    result = pyflagr_search(lattice)
    return result.predicted_keys

if __name__ == "__main__":
    preds = test_pyflagr()
    print("predicted counterfactual keys:", sorted(preds))
