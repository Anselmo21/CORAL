"""Generate a self-contained HTML viewer for data/gemini-created-dataset.json.

The output HTML embeds the dataset as a JS const, so it can be opened directly
in a browser (double-click) with no server required. Navigation: Prev/Next
buttons or left/right arrow keys; jump-to via the entry number input.
"""
from __future__ import annotations

import html
import json

DATA_PATH = "data/gemini-created-dataset.json"
OUT_PATH = "gemini_dataset_viewer.html"


def build_html(data: list[dict]) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    n = len(data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Gemini dataset viewer ({n} entries)</title>
<style>
  :root {{
    --fg: #1a1a1a;
    --muted: #6b6b6b;
    --bg: #ffffff;
    --card: #fafafa;
    --border: #e3e3e3;
    --accent: #2563eb;
    --accent-soft: #dbeafe;
    --pragma-temporal: #f59e0b;
    --pragma-geographic: #10b981;
    --pragma-trust: #8b5cf6;
    --pragma-perspective: #ec4899;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.55;
    color: var(--fg);
    background: var(--bg);
  }}
  .container {{
    max-width: 880px;
    margin: 0 auto;
    padding: 28px 24px 60px;
  }}
  .nav {{
    position: sticky;
    top: 0;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 14px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    z-index: 10;
  }}
  .nav button {{
    font-size: 16px;
    padding: 8px 14px;
    border: 1px solid var(--border);
    background: var(--card);
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
  }}
  .nav button:hover:not(:disabled) {{ background: var(--accent-soft); border-color: var(--accent); }}
  .nav button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .counter {{
    flex: 1;
    text-align: center;
    font-size: 15px;
    font-weight: 500;
    color: var(--muted);
  }}
  .counter input {{
    width: 50px;
    text-align: center;
    font-size: 15px;
    font-family: inherit;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 4px;
  }}
  h1 {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin: 28px 0 6px;
  }}
  .query {{
    font-size: 22px;
    font-weight: 600;
    line-height: 1.35;
    padding: 14px 18px;
    background: var(--accent-soft);
    border-left: 4px solid var(--accent);
    border-radius: 4px;
    margin: 6px 0 4px;
  }}
  .meta {{
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 18px;
  }}
  section {{ margin: 24px 0; }}
  section h2 {{
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin: 0 0 10px;
  }}
  .sources {{ display: flex; flex-direction: column; gap: 10px; }}
  .source-card {{
    border-left: 3px solid #94a3b8;
    background: var(--card);
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 15px;
  }}
  .source-card .src-label {{
    font-weight: 600;
    color: #475569;
    margin-right: 6px;
  }}
  .criteria, .pragmas {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 16px;
  }}
  .criterion, .pragma {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .criterion .idx, .pragma .idx {{
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--accent);
    color: white;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }}
  .pragma .tag {{
    flex-shrink: 0;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 8px;
    border-radius: 10px;
    color: white;
  }}
  .pragma .tag.temporal {{ background: var(--pragma-temporal); }}
  .pragma .tag.geographic {{ background: var(--pragma-geographic); }}
  .pragma .tag.trustworthiness {{ background: var(--pragma-trust); }}
  .pragma .tag.perspective {{ background: var(--pragma-perspective); }}
  .pragma .tag.other {{ background: var(--muted); }}
  .answers {{ display: flex; flex-direction: column; gap: 10px; }}
  .answer-card {{
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    background: var(--card);
    font-size: 15px;
  }}
  .answer-card .ans-label {{
    font-weight: 600;
    color: var(--accent);
    margin-right: 6px;
  }}
  kbd {{
    font-size: 11px;
    padding: 2px 6px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .help {{
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    margin-top: 6px;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="nav">
    <button id="prev">← Prev</button>
    <div class="counter">
      Entry <input type="number" id="jump" min="1" max="{n}" value="1" /> / {n}
    </div>
    <button id="next">Next →</button>
  </div>
  <div class="help">Use <kbd>←</kbd> / <kbd>→</kbd> arrow keys to navigate.</div>
  <div id="content"></div>
</div>
<script>
const DATA = {data_json};
let idx = 0;

function classifyPragma(p) {{
  const m = p.match(/^\\((\\w+)\\)\\s*:\\s*(.*)$/);
  if (!m) return {{ tag: 'Other', text: p, cls: 'other' }};
  const tag = m[1];
  const cls = tag.toLowerCase();
  return {{ tag, text: m[2], cls }};
}}

function esc(s) {{
  return String(s).replace(/[&<>"']/g, c => (
    {{ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }}[c]
  ));
}}

function render() {{
  const e = DATA[idx];
  const sources = e.retrieved_sources.map((s, i) => {{
    const m = s.match(/^(Source\\s+\\d+(?:\\s+\\([^)]*\\))?\\s*:)?\\s*(.*)$/);
    const label = m && m[1] ? m[1] : `Source ${{i+1}}:`;
    const body = m && m[1] ? m[2] : s;
    return `<div class="source-card"><span class="src-label">${{esc(label)}}</span>${{esc(body)}}</div>`;
  }}).join('');

  const criteria = e.criteria.map((c, i) =>
    `<div class="criterion"><span class="idx">${{i+1}}</span>${{esc(c)}}</div>`
  ).join('');

  const pragmas = e.pragmas.map(p => {{
    const {{ tag, text, cls }} = classifyPragma(p);
    return `<div class="pragma"><span class="tag ${{esc(cls)}}">${{esc(tag)}}</span>${{esc(text)}}</div>`;
  }}).join('');

  const answers = e.plausible_answers.map((a, i) =>
    `<div class="answer-card"><span class="ans-label">Answer ${{String.fromCharCode(65+i)}}</span>${{esc(a)}}</div>`
  ).join('');

  document.getElementById('content').innerHTML = `
    <h1>User query</h1>
    <div class="query">${{esc(e.user_query)}}</div>
    <div class="meta">id = ${{e.id}} • ${{e.retrieved_sources.length}} sources • ${{e.criteria.length}} criteria • ${{e.pragmas.length}} pragmas • ${{e.plausible_answers.length}} candidate answers</div>

    <section>
      <h2>Retrieved sources (${{e.retrieved_sources.length}})</h2>
      <div class="sources">${{sources}}</div>
    </section>

    <section>
      <h2>Criteria (${{e.criteria.length}})</h2>
      <div class="criteria">${{criteria}}</div>
    </section>

    <section>
      <h2>Pragmas (${{e.pragmas.length}})</h2>
      <div class="pragmas">${{pragmas}}</div>
    </section>

    <section>
      <h2>Plausible answers (${{e.plausible_answers.length}})</h2>
      <div class="answers">${{answers}}</div>
    </section>
  `;

  document.getElementById('prev').disabled = idx === 0;
  document.getElementById('next').disabled = idx === DATA.length - 1;
  document.getElementById('jump').value = idx + 1;
  window.scrollTo(0, 0);
}}

function go(delta) {{
  const next = Math.max(0, Math.min(DATA.length - 1, idx + delta));
  if (next !== idx) {{ idx = next; render(); }}
}}

document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);
document.getElementById('jump').addEventListener('change', (ev) => {{
  const v = parseInt(ev.target.value, 10);
  if (!isNaN(v)) {{ idx = Math.max(0, Math.min(DATA.length - 1, v - 1)); render(); }}
}});
document.addEventListener('keydown', (ev) => {{
  if (ev.target.tagName === 'INPUT') return;
  if (ev.key === 'ArrowLeft')  go(-1);
  if (ev.key === 'ArrowRight') go(1);
}});

render();
</script>
</body>
</html>
"""


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)
    html_out = build_html(data)
    with open(OUT_PATH, "w") as f:
        f.write(html_out)
    print(f"Wrote {OUT_PATH} ({len(data)} entries)")


if __name__ == "__main__":
    main()
