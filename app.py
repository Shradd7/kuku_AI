"""Promo Finder: prototype UI.

    streamlit run app.py

View mode   : open a finished run (outputs/ folder) instantly, with no API calls.
Run mode    : upload a .zip of episode screenplays and run the pipeline with the model in .env.
"""
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from promo.parse import parse_folder, fmt  # noqa: E402

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric, button, input {font-family: 'Figtree', sans-serif !important;}
.block-container {padding-top: 1.4rem; max-width: 1250px;}
/* Kuku TV palette: black / white / orange #E25506 / gold #FFBB10 */
.pf-brand {font-weight: 800; font-size: 1.05rem; letter-spacing: .02em;}
.pf-brand span {background: linear-gradient(346deg, #FF641C 0%, #C35F12 100%); -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;}
h1 {font-weight: 800 !important;}
[data-testid="stMetricValue"] {color: #FF641C; font-weight: 800;}
[data-testid="stMetricLabel"] {color: #9CA3AF;}
.pf-card {background: #111111; border: 1px solid rgba(255,255,255,.12); border-left: 4px solid #E25506;
          border-radius: 14px; padding: 14px 18px; margin-bottom: 8px;}
.pf-rank {font-size: 1.8rem; font-weight: 800; line-height: 1;
          background: linear-gradient(346deg, #FFBB10 0%, #B68511 100%); -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;}
.pf-tag {display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: .75rem; margin-right: 6px;
         background: rgba(255,255,255,.12); color: #E5E5E5;}
.pf-tag.score {background: linear-gradient(346deg, #FF641C 0%, #C35F12 100%); color: #fff; font-weight: 700;}
.pf-hook {font-weight: 700; font-size: 1.02rem;}
.pf-muted {color: #9CA3AF;}
.pf-line {font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .82rem; padding: 3px 8px;
          border-radius: 6px; color: #E5E5E5;}
.pf-first3 {background: rgba(226, 85, 6, .28); border-left: 3px solid #E25506;}
.pf-cliff {background: rgba(255, 187, 16, .18); border-left: 3px solid #FFBB10;}
.pf-gap {color: #9CA3AF; font-style: italic; font-size: .8rem; padding: 4px 8px;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.08);}
</style>
""", unsafe_allow_html=True)

SEG_RE = re.compile(r"E(\d+)\s+(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)")


def parse_segments(s):
    return [(int(e), int(m1) * 60 + float(s1), int(m2) * 60 + float(s2))
            for e, m1, s1, m2, s2 in SEG_RE.findall(str(s))]


@st.cache_data(show_spinner=False)
def load_run(out_dir):
    out = Path(out_dir)
    df = pd.read_csv(out / "promos.csv")
    log = json.loads((out / "run_log.json").read_text(encoding="utf-8"))
    qc_md = (out / "qc_report.md").read_text(encoding="utf-8") if (out / "qc_report.md").exists() else ""
    return df, log, qc_md


@st.cache_data(show_spinner=False)
def load_episodes(input_dir):
    try:
        eps = parse_folder(input_dir)
    except Exception:
        return {}
    return {e.number: e for e in eps}


def find_runs():
    runs = sorted({p.parent for p in ROOT.glob("outputs*/promos.csv")}, key=lambda p: p.name)
    return [str(p) for p in runs]


def script_html(eps, segs, hook_id, cliff_id):
    """The clip as a viewer would get it: dialogue/action lines, first 3s in yellow, cliffhanger in blue."""
    if not eps:
        return "<div class='pf-gap'>Script not available (input folder not found).</div>"
    rows = []
    t0 = segs[0][1] if segs else 0
    for k, (e, s, t) in enumerate(segs):
        ep = eps.get(e)
        if not ep:
            continue
        if k:
            rows.append(f"<div class='pf-gap'>✂ cut to E{e} {fmt(s)}</div>")
        for b in ep.blocks:
            if b.end <= s + 0.05 or b.start >= t - 0.05 or b.kind == "CAMERA":
                continue
            who = f"<b>{b.speaker}</b>" if b.kind == "DIALOGUE" else f"<i>{b.kind.lower()}</i>"
            cls = "pf-line"
            if k == 0 and b.start < t0 + 3.0:
                cls += " pf-first3"
            if b.id == cliff_id:
                cls += " pf-cliff"
            text = b.text if len(b.text) < 220 else b.text[:217] + "..."
            rows.append(f"<div class='{cls}'>{fmt(b.start)} {who}: {text}</div>")
    return "\n".join(rows)


def score_bar(label, v, invert=False):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return
    st.progress(min(1.0, max(0.0, (10 - v if invert else v) / 10)),
                text=f"{label}: {v:.0f}/10" + (" (lower is better)" if invert else ""))


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("<div class='pf-brand'>🎬 <span>PROMO FINDER</span></div>", unsafe_allow_html=True)
st.sidebar.caption("Finds 20-60s promo cuts in microdrama screenplays: hook in the first 3s, "
                   "cliffhanger at the end, makes sense on its own.")
mode = st.sidebar.radio("Mode", ["View a finished run", "Run on a new show"])

if mode == "Run on a new show":
    st.header("Run on a new show")
    up = st.file_uploader("Episode screenplays (.zip of .txt files)", type="zip")
    c1, c2, c3, c4 = st.columns(4)
    provider = c1.selectbox("Provider", ["(from .env)", "gemini", "openai", "anthropic", "offline"])
    model = c2.text_input("Model (blank = .env)", "")
    batch = c3.number_input("Episodes per call", 1, 10, 5)
    eps_range = c4.text_input("Episodes (blank = all)", "")
    if up and st.button("Find promos", type="primary"):
        work = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(up) as z:
            z.extractall(work / "in")
        out_dir = ROOT / f"outputs_{Path(up.name).stem[:30].replace(' ', '_')}"
        cmd = [sys.executable, "run.py", "--input", str(work / "in"), "--out", str(out_dir),
               "--show", Path(up.name).stem, "--batch", str(batch)]
        if provider != "(from .env)":
            cmd += ["--provider", provider]
        if model:
            cmd += ["--model", model]
        if eps_range:
            cmd += ["--episodes", eps_range]
        log_box = st.empty()
        lines = []
        with st.spinner("Reading episodes..."):
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-18:]))
            p.wait()
        if p.returncode == 0:
            st.success(f"Done. Switch to 'View a finished run' and pick `{out_dir.name}`.")
            load_run.clear()
        else:
            st.error("Run failed: see the log above.")
    st.stop()

# ------------------------------------------------------------------ view mode
runs = find_runs()
if not runs:
    st.info("No finished runs yet. Use 'Run on a new show', or run `python run.py ...` first.")
    st.stop()
default = next((i for i, r in enumerate(runs) if Path(r).name == "outputs"), 0)
run_dir = st.sidebar.selectbox("Run", runs, index=default, format_func=lambda r: Path(r).name)
df, log, qc_md = load_run(run_dir)
input_dir = st.sidebar.text_input("Script folder (for clip text)", log.get("input", ""))
eps = load_episodes(input_dir) if input_dir else {}

qc = log.get("qc", {})
pct = qc.get("pct", {})
lab = log.get("labels") or {}
st.title(log.get("show", "Promo Finder"))
st.caption(f"Model `{log.get('model')}` · prompt {log.get('prompt', 'v1')} · {log.get('episodes')} episodes · "
           f"{log.get('live_llm_calls', 0) + log.get('cached_llm_calls', 0)} LLM calls · "
           f"{log.get('tokens_in', 0):,} tokens in / {log.get('tokens_out', 0):,} out")

k = st.columns(5)
k[0].metric("Promos found", len(df))
k[1].metric("Within 20-60s", f"{pct.get('dur_ok', 0):.0f}%")
k[2].metric("Clean cuts", f"{pct.get('clean_cuts', 0):.0f}%", help="No cut lands inside a line of dialogue")
k[3].metric("Beyond ep. endings", f"{100 - pct.get('ends_at_episode_end', 0):.0f}%",
            help="Share of promos that do more than re-cut the writer's designed episode ending")
k[4].metric("Hand-picked found", f"{lab.get('found', '-')}/{lab.get('labels', '-')}" if lab else "-",
            help="Editor-picked reference promos covered at least 50% by a tool promo")

t1, t2, t3, t4, t5 = st.tabs(["⭐ Post these first", "All promos", "By episode", "Quality", "Raw model output"])

with t1:
    top = df[df["post_first"].notna() & (df["post_first"].astype(str) != "")].copy()
    top["post_first"] = top["post_first"].astype(float)
    top = top.sort_values("post_first")
    st.caption("Top picks, at most one per episode so each test covers a different story moment. "
               "Orange = first 3 seconds (the hook). Gold = the cliffhanger line it ends on.")
    for _, r in top.iterrows():
        segs = parse_segments(r["segments"])
        with st.container():
            st.markdown(
                f"<div class='pf-card'><span class='pf-rank'>#{int(r['post_first'])}</span> &nbsp;"
                f"<span class='pf-tag'>{r['promo_id']}</span><span class='pf-tag'>{r['duration_s']:.0f}s</span>"
                f"<span class='pf-tag'>{r['shape']}</span><span class='pf-tag score'>score {r['score']}</span><br>"
                f"<span class='pf-hook'>🪝 {r['hook']}</span><br>❓ {r['cliffhanger']}<br>"
                f"<small>✅ {r['why_it_works']}<br><span class='pf-muted'>⚠️ {r['weakness']}</span><br>"
                f"✂ <code>{r['segments']}</code></small></div>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                score_bar("Hook (first 3s)", r["hook_score"])
                score_bar("Cliffhanger", r["cliff_score"])
                score_bar("Makes sense alone", r["standalone"])
                score_bar("Spoiler risk", r["spoiler"], invert=True)
            with c2:
                with st.expander("Script of this cut", expanded=False):
                    st.markdown(script_html(eps, segs, r["hook_line_id"], r["cliff_line_id"]),
                                unsafe_allow_html=True)

with t2:
    c1, c2, c3 = st.columns(3)
    shapes = c1.multiselect("Shape", sorted(df["shape"].unique()), default=sorted(df["shape"].unique()))
    min_score = c2.slider("Min score", 0.0, 10.0, 0.0, 0.5)
    q = c3.text_input("Search hook / cliffhanger", "")
    view = df[df["shape"].isin(shapes) & (df["score"] >= min_score)]
    if q:
        view = view[view["hook"].str.contains(q, case=False, na=False) |
                    view["cliffhanger"].str.contains(q, case=False, na=False)]
    st.dataframe(view[["rank", "promo_id", "shape", "segments", "duration_s", "score", "hook", "cliffhanger",
                       "why_it_works"]], hide_index=True, width="stretch")
    st.download_button("Download promos.csv", df.to_csv(index=False).encode("utf-8"), "promos.csv")

with t3:
    d = df.copy()
    d["episode"] = d["segments"].map(lambda s: parse_segments(s)[0][0] if parse_segments(s) else None)
    n_eps = sorted(eps) if eps else range(1, int(d["episode"].max()) + 1)
    best = d.groupby("episode")["score"].max().reindex(n_eps).fillna(0)
    best.index = best.index.map(lambda n: f"E{int(n):02d}")
    st.caption("Best promo score per episode. Gaps = no promo passed the 20-60s / hook / cliffhanger rules.")
    st.bar_chart(best.rename("best promo score"), height=280, color="#E25506")
    st.caption("Promos per episode")
    cnt = d.groupby("episode").size().reindex(n_eps).fillna(0)
    cnt.index = cnt.index.map(lambda n: f"E{int(n):02d}")
    st.bar_chart(cnt.rename("promos"), height=180, color="#FFBB10")

with t4:
    st.markdown(qc_md or "_No QC report._")
    st.info("How to read this: rule checks prove the cuts are usable; 'hand-picked found' checks taste "
            "against an editor; 'ends at the episode's own ending' shows whether the tool finds anything "
            "beyond the writer's designed cliffhanger.")

with t5:
    raw_dir = Path(run_dir) / "raw_llm"
    files = sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []
    if not files:
        st.write("No raw model files for this run (offline baseline).")
    else:
        f = st.selectbox("Response file", files, format_func=lambda p: p.name)
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        st.caption(f"Model: {d.get('model')} · usage: {d.get('usage')}")
        c1, c2 = st.columns(2)
        with c1:
            st.text_area("Prompt (as sent)", d.get("prompt", ""), height=420)
        with c2:
            st.text_area("Raw response (unedited)", d.get("raw", ""), height=420)
