"""
Generate pragmas for the MSMARCO base dataset using gpt-4o-mini.

Uses extraction-based pragma prompting: temporal, geographic, perspective, trustworthiness
are extracted from the retrieved passages rather than generated from scratch.

Usage: OPENAI_API_KEY=sk-... python generate_msmarco_pragmas.py

Reads/writes datasets/msmarco-data.json in this schema:

  {
    "version": 1,
    "description": "...",
    "pragmas_model": "openai:gpt-4o-mini",
    "criteria_model": "...",
    "rows": [
      {
        "query_id": int,
        "query": str,
        "sources": str,
        "criteria": [str, ...],
        "pragmas": {"temporal": [...], "geographic": [...],
                    "perspective": [...], "trustworthiness": [...]},
        "plausible_answers": [...]
      },
      ...
    ]
  }

Resume-safe: re-running skips rows whose "pragmas" field is already a populated
dict. Pass --force to regenerate every row.
"""

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

client = OpenAI()

DATASET_PATH = Path("datasets/msmarco-data.json")
PRAGMA_MODEL = "gpt-4o-mini"
MAX_ATTEMPTS = 3

PRAGMA_DESCRIPTIONS = {
    "temporal": (
        "- **Temporal**: Relating to *when a source was published or when its information is valid* "
        "(e.g., publication year, a date range the source covers). Also include specific periods or "
        "events that mark inflection points—moments when the answer to the user's question could "
        "change (e.g., 'before Policy X (2021)', 'after Product Y launch (2023)'). "
        "Do NOT extract durations, frequencies, or time references that appear inside the content "
        "as instructions or facts (e.g., 'every 3 weeks', 'within 30 minutes')."
    ),
    "geographic": "- **Geographic**: Relating to place or origin.",
    "trustworthiness": (
        "- **Trustworthiness**: Pertaining to the reliability/standing of the publishers or "
        "sources. Use short descriptive labels — never output raw URLs or domain names. "
        "Infer the label from the domain when no explicit attribution is given "
        "(e.g., wikihow.com → 'how-to site', .gov → 'government source', "
        "answers.yahoo.com → 'user-generated Q&A', webmd.com → 'mainstream health site')."
    ),
    "perspective": (
        "- **Perspective**: Referring to side, slant, or degree of polarization shown in any "
        "source (e.g., 'left-leaning', 'neutral', 'opposition opinion')."
    ),
}

PRAGMA_TYPES = list(PRAGMA_DESCRIPTIONS.keys())


def build_pragma_prompt(question: str, sources_text: str) -> str:
    prompt_sources = f"SOURCE 1\n{sources_text.strip()}\n"

    prompt_pragmas = ""
    for desc in PRAGMA_DESCRIPTIONS.values():
        prompt_pragmas += desc + "\n"

    instructions = f"""
**Instructions:**
- Read the user's question, every provided source text, and every URL.
- URLs are valuable signals: domain names indicate trustworthiness (e.g., wikihow.com = how-to site, .gov = government source, yahoo answers = user-generated), and URL paths or query strings sometimes encode dates (e.g., /2023/03/ or ?date=20080225).
- Carefully extract each pragma type from all sources using only information explicitly present or logically inferred.
- There are four pragma types: temporal, geographic, perspective, trustworthiness.
- For each of the four pragma types, compile a deduplicated list of all distinct pragmas found across all sources (not separated by source).
- The total number of pragmas across all four types must be exactly 10.
- First extract what is genuinely present in the sources. If the total falls below 10, generate plausible additional pragmas for any underrepresented type that would meaningfully filter the source pool for this question.
- Make sure the pragmas are not semantically redundant.
- For **temporal**, only include publication dates/years of sources (including dates inferred from URLs) and inflection-point periods where the answer to the question could change. Do NOT include durations, frequencies, or time-based instructions found inside source content (e.g., "every 3 weeks", "within 30 minutes").
- If there are key historical events, turning points, or publication dates that distinguish perspectives or facts relevant to the user's question, include those moments as temporal pragmas.
- Pragmas should be as short and precise as possible (typically 2-6 words).
- Never generate pragmas that would directly answer or partially answer the question.
- Do not include information that cannot be reasonably inferred from the texts.
- Output the results in the structured JSON format described below.
- Output a JSON object only. Do not include Markdown code fences, backticks, or extra commentary.
"""

    output_format = f"""
**Output Format:**
Output a strict JSON object with exactly 4 fields:

"""
    if "temporal" in PRAGMA_TYPES:
        output_format += "**temporal**: Array of all unique temporal pragmas found (including inflection points where the answer might change).\n"
    if "geographic" in PRAGMA_TYPES:
        output_format += "**geographic**: Array of all unique geographic pragmas found.\n"
    if "perspective" in PRAGMA_TYPES:
        output_format += "**perspective**: Array of all unique perspective pragmas found.\n"
    if "trustworthiness" in PRAGMA_TYPES:
        output_format += "**trustworthiness**: Array of all unique trustworthiness pragmas found.\n"
    output_format += "\nEach field must be an array (even if empty). Do not include pragmas redundantly.\n"

    example = """
# Example

#### Input
- User Question: "What caused the recent protests in Paris?"
- Source 1: "Published March 2023 by Le Monde, a Paris-based newspaper, with a centrist editorial stance. The article primarily covers perspectives from protest organizers."
- Source 2: "Published by the Interior Ministry of France on March 6, 2023. Presents the official government position."
- Source 3: "An editorial published February 2023 describes events before the pension reform was passed."

#### Output
{
"temporal": ["March 2023", "March 6, 2023", "February 2023", "before pension reform passed (February 2023)", "after pension reform passed (March 2023)"],
"geographic": ["Paris", "France"],
"perspective": ["centrist", "protest organizers' view", "government position"],
"trustworthiness": ["Le Monde", "mainstream newspaper", "Interior Ministry", "official source"]
}
"""

    note = """
# Notes

- For temporal pragmas: Only use publication dates/years of sources and inflection-point periods (when the answer could change). Never use durations, frequencies, or time references that appear as instructions or facts within source content.
- Do not include any pragmas that directly address or imply an answer to the user's question.
- Each pragma should appear only once in its respective array.

**Objective restated:** Extract and organize all distinct pragmas by pragma type (temporal—including key inflection points—, geographic, perspective, trustworthiness) found across all sources, and return them together in a single structured JSON object, not grouped per source. Do not answer or imply an answer to the user's question.
"""

    return (
        "Extract all relevant pragmatic contextual terms—found in the provided sources connected to a user's question, "
        "and group these by pragma type (not by source). Pragmas must be concise, accurate, and must never address or "
        "attempt to answer the user's question.\n"
        "There are the following types of pragmas you can extract:\n"
        f"{prompt_pragmas}"
        f"{instructions}\n"
        f"{output_format}\n"
        f"{example}\n"
        f"QUESTION\n{question}\n\n"
        f"{prompt_sources}\n\n"
        f"{note}"
    )


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    inner = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    inner = inner.strip()
    if inner.lower().startswith("json"):
        inner = inner[4:].strip()
    return inner


def _extract_json_object(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_pragmas_response(text: str) -> dict:
    payload = json.loads(_extract_json_object(text))
    for key in PRAGMA_TYPES:
        if key not in payload:
            raise ValueError(f'Missing key "{key}" in response')
        if not isinstance(payload[key], list):
            raise ValueError(f'"{key}" must be an array, got {type(payload[key]).__name__}')
    result = {key: [str(p).strip() for p in payload[key]] for key in PRAGMA_TYPES}
    total = sum(len(v) for v in result.values())
    if total != 10:
        raise ValueError(f"Expected 10 pragmas total across all types, got {total}")
    return result


def generate_pragmas(question: str, sources: str) -> dict:
    user_content = build_pragma_prompt(question, sources)
    messages = [{"role": "user", "content": user_content}]
    last_error = ""
    last_text = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"},
            messages=messages,
        )
        last_text = (resp.choices[0].message.content or "").strip()
        try:
            return _parse_pragmas_response(last_text)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
            if attempt == MAX_ATTEMPTS:
                break
            messages.append({"role": "assistant", "content": last_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid JSON or did not follow the schema.\n"
                        f"Reason: {last_error}\n"
                        f"Return ONLY a corrected JSON object with exactly these keys: {PRAGMA_TYPES}. "
                        "Each value must be an array of strings."
                    ),
                }
            )
            time.sleep(0.5)

    preview = re.sub(r"\s+", " ", last_text)[:240]
    raise RuntimeError(
        f"Failed to parse pragmas after {MAX_ATTEMPTS} attempts: {last_error}. Last response: {preview!r}"
    )


def _has_pragmas(row: dict) -> bool:
    p = row.get("pragmas")
    if not isinstance(p, dict):
        return False
    return any(p.get(k) for k in PRAGMA_TYPES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=DATASET_PATH,
                    help=f"Path to the msmarco base dataset (default: {DATASET_PATH})")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate pragmas even for rows that already have them.")
    args = ap.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    rows = data["rows"]
    total = len(rows)

    for i, row in enumerate(rows, 1):
        qid = row["query_id"]
        if not args.force and _has_pragmas(row):
            print(f"[{i:02d}/{total}] {qid}: skip (pragmas present)")
            continue
        print(f"[{i:02d}/{total}] {qid}: {row['query'][:60]}...")
        try:
            row["pragmas"] = generate_pragmas(row["query"], row["sources"])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        data["pragmas_model"] = f"openai:{PRAGMA_MODEL}"
        args.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        time.sleep(0.3)

    done = sum(1 for r in rows if _has_pragmas(r))
    print(f"\nDone. {done}/{total} rows have pragmas.")


if __name__ == "__main__":
    main()
