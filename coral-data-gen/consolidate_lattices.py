"""
Consolidate per-qid lattice outputs under lattices/<run-name>/ into one JSON file.

Each qid directory has config.json + answers.json. Output matches the
data-generation consolidated format: _meta block plus qid-keyed entries with
raw_answers (compatible with DatasetLoader).

Usage:
    python consolidate_lattices.py                          # all runs under lattices/
    python consolidate_lattices.py --run-name sonnet-criteria-qwen3-32b
    python consolidate_lattices.py --run-name msmarco-pragmas-qwen3-32b --output-dir out/
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LATTICES_ROOT = Path("lattices")
DEFAULT_OUTPUT_DIR = Path("datasets/merged")


def _qid_sort_key(name: str) -> tuple:
    return (0, int(name)) if name.isdigit() else (1, name)


def _parse_run_name(run_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (dataset, source) from run names like sonnet-criteria-qwen3-32b."""
    for source in ("criteria", "pragmas"):
        marker = f"-{source}-"
        if marker in run_name:
            return run_name.split(marker, 1)[0], source
    return None, None


def _summarize(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    complete = halted = partial = 0
    for entry in entries:
        if entry.get("halted"):
            halted += 1
        elif entry.get("combos_run") == entry.get("combos_total"):
            complete += 1
        else:
            partial += 1
    return {
        "queries": len(entries),
        "complete": complete,
        "halted": halted,
        "partial": partial,
    }


def consolidate_run(
    run_dir: Path,
    *,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Consolidate one lattices/<run-name>/ tree into a qid-keyed dataset dict."""
    run_name = run_dir.name
    dataset, source = _parse_run_name(run_name)
    entries: List[Dict[str, Any]] = []

    for qid_dir in sorted(
        (p for p in run_dir.iterdir() if p.is_dir()),
        key=lambda p: _qid_sort_key(p.name),
    ):
        config_path = qid_dir / "config.json"
        answers_path = qid_dir / "answers.json"
        if not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        answers: Dict[str, Any] = {}
        if answers_path.exists():
            try:
                answers = json.loads(answers_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        qid = qid_dir.name
        entry = dict(config)
        entry["name"] = qid
        entry["raw_answers"] = answers
        entries.append(entry)

    mode = entries[0].get("mode") if entries else source
    summary = _summarize(entries)

    consolidated: Dict[str, Any] = {
        "_meta": {
            "version": 1,
            "description": description or (
                f"Consolidated lattice run from lattices/{run_name}/ "
                f"({summary['queries']} queries, mode={mode})"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "dataset": dataset,
            "source": source or mode,
            "mode": mode,
            "summary": summary,
        }
    }

    for entry in entries:
        consolidated[entry["name"]] = entry

    return consolidated


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--lattices-root", type=Path, default=LATTICES_ROOT)
    ap.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="Run dir name under lattices/ (repeatable; default: all)",
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--description",
        default=None,
        help="Override _meta.description (default: auto from run name)",
    )
    args = ap.parse_args()

    root = args.lattices_root
    if not root.is_dir():
        raise SystemExit(f"Lattices root not found: {root}")

    if args.run_name:
        run_dirs = [root / name for name in args.run_name]
    else:
        run_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"Skip missing run: {run_dir}")
            continue

        consolidated = consolidate_run(run_dir, description=args.description)
        out_path = args.output_dir / f"{run_dir.name}.json"
        out_path.write_text(
            json.dumps(consolidated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        s = consolidated["_meta"]["summary"]
        print(
            f"Wrote {out_path} — {s['queries']} queries "
            f"({s['complete']} complete, {s['halted']} halted, {s['partial']} partial)"
        )


if __name__ == "__main__":
    main()
