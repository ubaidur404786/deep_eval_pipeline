"""
app/ui_theme.py -- the demo's visual identity, kept out of the app logic.

    inject_css()                  fonts, colours, spacing for Streamlit's widgets
    failure_map_html(runs, ...)   the 26-cases x N-models pass/fail grid (the hero)

Colour has meaning here and nowhere else:
    pass       #1E7F6B     every metric for the case passed
    fail       #B3261E     at least one metric failed
    retrieval  #8A6D1F     the retriever missed an expected article (a hollow cell)
Everything interactive is one blue (#2457C5). Everything else is ink on paper.
"""

import html
import json
from pathlib import Path

import streamlit as st

INK = "#1B2A3A"
PAPER = "#F3F6F4"
PAPER_2 = "#E8EDEA"
COBALT = "#2457C5"
PASS = "#1E7F6B"
FAIL = "#B3261E"
RETRIEVAL = "#8A6D1F"
MUTED = "#5B6B78"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {{
  --ink: {INK}; --paper: {PAPER}; --paper-2: {PAPER_2}; --cobalt: {COBALT};
  --pass: {PASS}; --fail: {FAIL}; --retrieval: {RETRIEVAL}; --muted: {MUTED};
  --display: 'Bricolage Grotesque', 'Instrument Sans', system-ui, sans-serif;
  --text: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}

/* ground: paper with a faint 24px grid -- a measurement surface */
.stApp {{
  background-color: var(--paper);
  background-image:
    linear-gradient(rgba(27,42,58,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(27,42,58,.055) 1px, transparent 1px);
  background-size: 24px 24px;
}}
/* Fonts are set through .streamlit/config.toml (theme.font / headingFont), so
   Streamlit applies them to its own text and leaves its icon font alone.
   The @import above only makes the faces available. */
.stApp {{ color: var(--ink); }}
/* let the grid run to the top */
header[data-testid="stHeader"] {{ background: transparent !important; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  letter-spacing: -0.015em;
  color: var(--ink);
}}
.stApp h1 {{ font-size: clamp(2.1rem, 4.2vw, 3.4rem); font-weight: 800; line-height: 1.02; margin: .2rem 0 .4rem; max-width: 18ch; text-wrap: balance; }}
.stApp h4 {{ font-size: 1.05rem; font-weight: 700; margin-top: 1.4rem; }}

/* the main column: measured width, left aligned, comfortable line length */
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }}

/* hero */
.hero-lede {{
  font-family: var(--text); font-size: 1.12rem; line-height: 1.5; color: var(--ink);
  max-width: 62ch; margin: 0 0 .35rem;
}}
.hero-meta {{ font-size: .92rem; color: var(--muted); margin: 0 0 1.6rem; }}
.hero-meta a {{ color: var(--cobalt); text-decoration: underline; text-underline-offset: 2px; }}

/* failure map */
.fmap {{ margin: .4rem 0 .6rem; }}
.fmap-row {{ display: grid; grid-template-columns: 11.5rem 1fr 5.2rem; align-items: center; gap: .8rem; padding: .3rem 0; }}
.fmap-row + .fmap-row {{ border-top: 1px solid rgba(27,42,58,.10); }}
.fmap-label {{ font-size: .93rem; font-weight: 600; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.fmap-cells {{ display: grid; grid-template-columns: repeat(26, minmax(0, 1fr)); gap: 4px; }}
.fmap-cell {{
  display: block; aspect-ratio: 1 / 1; border-radius: 3px; border: 2px solid transparent;
  background: var(--pass); transition: transform .12s ease;
}}
.fmap-cell.fail {{ background: var(--fail); }}
.fmap-cell.retrieval {{ background: transparent; border-color: var(--retrieval); }}
.fmap-cell.retrieval.fail {{ background: rgba(179,38,30,.35); border-color: var(--retrieval); }}
.fmap-cell:hover, .fmap-cell:focus-visible {{ transform: scale(1.35); outline: 2px solid var(--cobalt); outline-offset: 1px; z-index: 2; position: relative; }}
.fmap-count {{ font-variant-numeric: tabular-nums; font-size: .93rem; text-align: right; color: var(--ink); }}
.fmap-count b {{ font-family: var(--display); font-weight: 800; font-size: 1.15rem; }}
.fmap-axis {{ display: grid; grid-template-columns: 11.5rem 1fr 5.2rem; gap: .8rem; font-size: .78rem; color: var(--muted); margin-top: .35rem; }}
.fmap-axis .ticks {{ display: grid; grid-template-columns: repeat(26, minmax(0, 1fr)); gap: 4px; }}
.fmap-axis .ticks span {{ text-align: center; font-variant-numeric: tabular-nums; }}
.fmap-legend {{ display: flex; flex-wrap: wrap; gap: 1.1rem; font-size: .88rem; color: var(--muted); margin: .6rem 0 0; align-items: center; }}
.fmap-legend i {{ display: inline-block; width: .8rem; height: .8rem; border-radius: 2px; vertical-align: -1px; margin-right: .4rem; }}
.fmap-legend .i-pass {{ background: var(--pass); }}
.fmap-legend .i-fail {{ background: var(--fail); }}
.fmap-legend .i-ret {{ border: 2px solid var(--retrieval); box-sizing: border-box; }}
.fmap-hint {{ color: var(--muted); }}
@media (max-width: 720px) {{
  .fmap-row {{ grid-template-columns: 1fr; gap: .3rem; }}
  .fmap-axis {{ display: none; }}
  .fmap-count {{ text-align: left; }}
}}

/* Streamlit chrome, quieted */
[data-testid="stSidebar"] {{ background: var(--paper-2); border-right: 1px solid rgba(27,42,58,.12); }}
[data-testid="stSidebar"] h2 {{ font-size: 1.15rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid rgba(27,42,58,.18); }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: .98rem; padding: .6rem 0; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--cobalt); height: 3px; }}
.stButton > button {{
  border-radius: 4px; font-weight: 600; border: 1.5px solid var(--ink); background: transparent; color: var(--ink);
}}
.stButton > button:hover, .stButton > button:focus-visible {{ border-color: var(--cobalt); color: var(--cobalt); background: rgba(36,87,197,.06); }}
.stButton > button:focus-visible {{ outline: 3px solid rgba(36,87,197,.35); outline-offset: 2px; }}
[data-testid="stMetric"] {{ background: transparent; border-left: 3px solid rgba(27,42,58,.18); padding: .2rem 0 .2rem .9rem; }}
[data-testid="stMetricValue"] {{ font-family: var(--display); font-weight: 800; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stExpander"] {{ border: 1px solid rgba(27,42,58,.14); border-radius: 4px; background: rgba(255,255,255,.55); }}
.stAlert {{ border-radius: 4px; }}
[data-testid="stDataFrame"] {{ border: 1px solid rgba(27,42,58,.14); border-radius: 4px; }}
code {{ color: var(--ink) !important; font-variant-numeric: tabular-nums; background: rgba(27,42,58,.06); border-radius: 3px; padding: .05em .3em; }}

@media (prefers-reduced-motion: reduce) {{
  .fmap-cell {{ transition: none; }}
  .fmap-cell:hover {{ transform: none; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, lede: str, meta_html: str) -> None:
    st.markdown(f"# {title}")
    st.markdown(f'<p class="hero-lede">{html.escape(lede)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="hero-meta">{meta_html}</p>', unsafe_allow_html=True)


def failure_map_html(runs: list[tuple[str, dict]], case_ids: list[str]) -> str:
    """runs: [(label, results_dict), ...] sharing the same judge. One row per run,
    one cell per case. Each cell links to ?case=<id> so a click opens that case."""
    rows = []
    for label, r in runs:
        cases = {c["id"]: c for c in r["cases"]}
        cells = []
        n_fail = 0
        for cid in case_ids:
            c = cases.get(cid)
            if c is None:
                cells.append('<span class="fmap-cell" style="background:transparent;border-color:rgba(27,42,58,.2)"></span>')
                continue
            failed = [m["metric"] for m in c["metrics"] if not m["passed"]]
            src = c.get("expected_sources_found")
            retrieval_miss = bool(src) and src.split("/")[0] != src.split("/")[1]
            classes = "fmap-cell" + (" fail" if failed else "") + (" retrieval" if retrieval_miss else "")
            n_fail += bool(failed)
            tip = f"{cid} · {c['category']}"
            tip += f" · failed: {', '.join(failed)}" if failed else " · passed"
            if retrieval_miss:
                tip += f" · retriever found {src} expected articles"
            cells.append(f'<a class="{classes}" href="?case={cid}" target="_self" title="{html.escape(tip)}" aria-label="{html.escape(tip)}"></a>')
        rows.append(
            f'<div class="fmap-row"><div class="fmap-label" title="{html.escape(label)}">{html.escape(label)}</div>'
            f'<div class="fmap-cells">{"".join(cells)}</div>'
            f'<div class="fmap-count"><b>{n_fail}</b> failed</div></div>'
        )
    ticks = "".join(f"<span>{i if i in (1, 5, 10, 15, 20, 26) else ''}</span>" for i in range(1, len(case_ids) + 1))
    axis = f'<div class="fmap-axis"><div></div><div class="ticks">{ticks}</div><div></div></div>'
    legend = (
        '<div class="fmap-legend">'
        '<span><i class="i-pass"></i>all metrics passed</span>'
        '<span><i class="i-fail"></i>a metric failed</span>'
        '<span><i class="i-ret"></i>retriever missed an expected article</span>'
        '<span class="fmap-hint">Click a cell to open that case below.</span>'
        '</div>'
    )
    return f'<div class="fmap">{"".join(rows)}{axis}{legend}</div>'
