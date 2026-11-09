"""Promo Finder: prototype UI.     streamlit run app.py

View   : open a finished run (outputs*/ folder) instantly, with no API calls.
Run    : upload a .zip of episode screenplays and run the pipeline with the model in .env.
"""
import html
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

st.set_page_config(page_title="Promo Finder", layout="wide")

# ---------------------------------------------------------------- design system (Kuku FM palette)
# colour: black canvas, two surfaces, one accent (Kuku red), gold only to mark the cliffhanger
# type  : Figtree, 13 / 15 / 18 / 24 / 36 px     space: 8 / 16 / 24 / 32 / 48     radius: 16, pills 999
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#000; --s1:#141414; --s2:#1F1F1F; --line:rgba(255,255,255,.08);
  --tx:#FFFFFF; --tx2:#B4B4BB; --tx3:#7C7C85;
  --accent:#EB0000; --accent-soft:rgba(235,0,0,.16); --gold:#FFBB10; --gold-soft:rgba(255,187,16,.14);
  --r:16px;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select, label {
  font-family:'Figtree',system-ui,sans-serif !important; }
.stApp{background:var(--bg);}
.block-container{padding:56px 48px 64px; max-width:1180px;}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {display:none !important;}
section[data-testid="stSidebar"]{background:var(--s1); border-right:1px solid var(--line);}

/* type scale */
.t-eyebrow{font-size:13px; font-weight:600; color:var(--tx3); letter-spacing:.02em; margin:0 0 8px;}
.t-title{font-size:36px; font-weight:800; line-height:1.1; color:var(--tx); margin:0 0 8px;}
.t-meta{font-size:13px; color:var(--tx3); margin:0;}
.t-h2{font-size:24px; font-weight:700; color:var(--tx); line-height:1.25; margin:0;}
.t-body{font-size:15px; color:var(--tx2); line-height:1.55; margin:0;}
.t-small{font-size:13px; color:var(--tx3);}

/* stats row */
.stats{display:flex; gap:48px; margin:32px 0 8px; flex-wrap:wrap;}
.stat .v{font-size:28px; font-weight:800; color:var(--tx); line-height:1;}
.stat .l{font-size:13px; color:var(--tx3); margin-top:6px;}

/* hero: the #1 promo, the product itself */
.hero{background:var(--s1); border-radius:var(--r); padding:32px; margin:24px 0 16px;
      display:grid; grid-template-columns: 1fr 1.15fr; gap:32px;}
.hero .rank{font-size:13px; font-weight:700; color:var(--accent); margin-bottom:16px;}
.hero .hook{font-size:24px; font-weight:700; line-height:1.3; color:var(--tx); margin-bottom:16px;}
.hero .ends{font-size:15px; color:var(--tx2); line-height:1.5; margin-bottom:24px;}
.hero .ends b{color:var(--gold); font-weight:600;}
.kv{display:grid; grid-template-columns:auto 1fr; gap:8px 16px; font-size:13px;}
.kv .k{color:var(--tx3);} .kv .v{color:var(--tx2);}
.scores{display:flex; gap:24px; margin:24px 0 0;}
.scores div{font-size:13px; color:var(--tx3);} .scores b{display:block; font-size:18px; color:var(--tx); font-weight:700;}

/* the clip script */
.clip{background:var(--bg); border-radius:12px; padding:16px; max-height:430px; overflow:auto;}
.clip .ln{display:grid; grid-template-columns:52px 1fr; gap:12px; padding:6px 8px; border-radius:8px;
          font-size:14px; line-height:1.45; color:var(--tx2);}
.clip .ts{font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; color:var(--tx3); padding-top:2px;}
.clip .who{color:var(--tx); font-weight:600;}
.clip .act{color:var(--tx3); font-style:italic;}
.clip .hk{background:var(--accent-soft);} .clip .cl{background:var(--gold-soft);}
.clip .cut{font-size:12px; color:var(--tx3); padding:8px; text-align:center;}
.legend{font-size:12px; color:var(--tx3); margin:12px 0 0;}
.legend i{display:inline-block; width:10px; height:10px; border-radius:3px; margin:0 6px 0 12px; vertical-align:-1px;}

/* compact rows for picks 2..10 */
.row{display:grid; grid-template-columns:40px 1fr 140px; gap:16px; align-items:start; padding:16px 0;
     border-top:1px solid var(--line);}
.row .n{font-size:18px; font-weight:700; color:var(--tx3);}
.row .h{font-size:16px; font-weight:600; color:var(--tx); line-height:1.35;}
.row .e{font-size:14px; color:var(--tx2); margin-top:4px;}
.row .m{font-size:13px; color:var(--tx3); text-align:right; line-height:1.6;}

/* controls */
.stButton button, .stDownloadButton button{border-radius:999px !important; font-weight:600 !important;
   padding:10px 22px !important;}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"]{background:var(--accent) !important;
   border-color:var(--accent) !important; color:#fff !important;}
[data-baseweb="tab-list"]{gap:24px;}
[data-baseweb="tab"]{font-size:15px !important; padding:8px 0 !important;}
[data-baseweb="tab-highlight"]{background:var(--accent) !important;}
details{border:none !important; background:var(--s1); border-radius:12px !important;}
</style>
""", unsafe_allow_html=True)

SEG_RE = re.compile(r"E(\d+)\s+(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)")
esc = html.escape


def parse_segments(s):
    return [(int(e), int(m1) * 60 + float(s1), int(m2) * 60 + float(s2))
            for e, m1, s1, m2, s2 in SEG_RE.findall(str(s))]


def cut_text(segs):
    """'Episode 3 · 1:24–2:15' (and '+ Episode 4 · …' for montages)."""
    def t(x):
        m, s = divmod(int(round(x)), 60)
        return f"{m}:{s:02d}"
    return " + ".join(f"Episode {e} · {t(a)}–{t(b)}" for e, a, b in segs)


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
        return {e.number: e for e in parse_folder(input_dir)}
    except Exception:
        return {}


def find_runs():
    return sorted({str(p.parent) for p in ROOT.glob("outputs*/promos.csv")}, key=lambda p: Path(p).name)


def clip_html(eps, segs, hook_id, cliff_id):
    if not eps:
        return "<div class='clip'><span class='t-small'>Point the sidebar at the script folder to see the clip.</span></div>"
    out, t0 = [], segs[0][1] if segs else 0
    for k, (e, s, t) in enumerate(segs):
        ep = eps.get(e)
        if not ep:
            continue
        if k:
            out.append(f"<div class='cut'>cut to episode {e}</div>")
        for b in ep.blocks:
            if b.end <= s + 0.05 or b.start >= t - 0.05 or b.kind == "CAMERA":
                continue
            cls = "ln"
            if (k == 0 and b.start < t0 + 3.0) or b.id == hook_id:
                cls += " hk"
            if b.id == cliff_id:
                cls += " cl"
            if b.kind == "DIALOGUE":
                body = f"<span class='who'>{esc(b.speaker.title())}</span> &nbsp;{esc(b.text)}"
            else:
                body = f"<span class='act'>{esc(b.text)}</span>"
            out.append(f"<div class='{cls}'><span class='ts'>{fmt(b.start)[:-3]}</span><span>{body}</span></div>")
    return "<div class='clip'>" + "".join(out) + "</div>"


def level(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    return "Low" if v <= 3 else ("Medium" if v <= 6 else "High")


def num(v):
    try:
        return f"{float(v):.0f}"
    except (TypeError, ValueError):
        return "–"


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("<div class='t-h2' style='font-size:18px'>Promo Finder</div>"
                    "<div class='t-small' style='margin:4px 0 24px'>Promo cuts from microdrama scripts</div>",
                    unsafe_allow_html=True)
mode = st.sidebar.radio("Mode", ["View results", "Run on a new show"], label_visibility="collapsed")

# ---------------------------------------------------------------- run mode
if mode == "Run on a new show":
    st.markdown("<div class='t-eyebrow'>New show</div><div class='t-title'>Find promos in a new show</div>"
                "<div class='t-body'>Upload a .zip with one screenplay .txt per episode. "
                "The model comes from your .env file.</div>", unsafe_allow_html=True)
    st.write("")
    up = st.file_uploader("Screenplays (.zip)", type="zip")
    c1, c2, c3 = st.columns(3)
    provider = c1.selectbox("Provider", ["From .env", "gemini", "openai", "anthropic", "offline"])
    model = c2.text_input("Model", "", placeholder="From .env")
    eps_range = c3.text_input("Episodes", "", placeholder="All, or e.g. 1-5")
    batch = st.slider("Episodes per model call", 1, 10, 5,
                      help="5 keeps a 50-episode show to about 15 calls, inside free daily quotas.")
    st.write("")
    go = st.button("Find promos", type="primary", disabled=up is None)
    if up and go:
        work = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(up) as z:
            z.extractall(work / "in")
        out_dir = ROOT / f"outputs_{re.sub(r'[^A-Za-z0-9]+', '_', Path(up.name).stem)[:30]}"
        cmd = [sys.executable, "run.py", "--input", str(work / "in"), "--out", str(out_dir),
               "--show", Path(up.name).stem, "--batch", str(batch)]
        if provider != "From .env":
            cmd += ["--provider", provider]
        if model:
            cmd += ["--model", model]
        if eps_range:
            cmd += ["--episodes", eps_range]
        box, lines = st.empty(), []
        with st.spinner("Reading episodes"):
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                lines.append(line.rstrip())
                box.code("\n".join(lines[-14:]), language=None)
            p.wait()
        if p.returncode == 0:
            load_run.clear()
            st.success(f"Done. Open View results and pick “{out_dir.name}”.")
        else:
            st.error("The run stopped. The log above shows why.")
    st.stop()

# ---------------------------------------------------------------- view mode
runs = find_runs()
if not runs:
    st.info("No results yet. Use “Run on a new show”, or run run.py first.")
    st.stop()
pref = [i for n in ("outputs_v4_lite", "outputs_v31", "outputs") for i, r in enumerate(runs) if Path(r).name == n]
run_dir = st.sidebar.selectbox("Results", runs, index=pref[0] if pref else 0, format_func=lambda r: Path(r).name)
df, log, qc_md = load_run(run_dir)
with st.sidebar.expander("Script folder"):
    input_dir = st.text_input("Path", log.get("input", ""), label_visibility="collapsed")
eps = load_episodes(input_dir) if input_dir else {}
st.sidebar.markdown(f"<div class='t-small' style='margin-top:24px'>Model {esc(str(log.get('model', '')).split(':')[-1])}"
                    f"<br>Prompt {esc(str(log.get('prompt', 'v1')))} · assembly {esc(str(log.get('assembly', 'v3')))}"
                    f"<br>{log.get('live_llm_calls', 0) + log.get('cached_llm_calls', 0)} model calls · "
                    f"{log.get('tokens_in', 0) // 1000}k tokens in</div>", unsafe_allow_html=True)

qc, lab = log.get("qc", {}), (log.get("labels") or {})
pct = qc.get("pct", {})
top = df[df["post_first"].notna() & (df["post_first"].astype(str) != "")].copy()
top["post_first"] = top["post_first"].astype(float)
top = top.sort_values("post_first")

st.markdown(
    f"<div class='t-eyebrow'>Promo report</div><div class='t-title'>{esc(log.get('show', 'Untitled show'))}</div>"
    f"<div class='t-meta'>{log.get('episodes')} episodes read · {len(df)} promos found · "
    f"top {len(top)} ranked to post first</div>"
    f"<div class='stats'>"
    f"<div class='stat'><div class='v'>{len(df)}</div><div class='l'>Promos, all 20–60s</div></div>"
    f"<div class='stat'><div class='v'>{100 - pct.get('ends_at_episode_end', 0):.0f}%</div>"
    f"<div class='l'>Find moments beyond the episode ending</div></div>"
    f"<div class='stat'><div class='v'>{lab.get('found', '–')}/{lab.get('labels', '–')}</div>"
    f"<div class='l'>Editor-picked promos matched</div></div>"
    f"<div class='stat'><div class='v'>{pct.get('clean_cuts', 0):.0f}%</div><div class='l'>Cuts on a line boundary</div></div>"
    f"</div>", unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs(["Top picks", "All promos", "By episode", "Quality", "Model output"])

with t1:
    if top.empty:
        st.write("No ranked promos in this run.")
    else:
        r = top.iloc[0]
        segs = parse_segments(r["segments"])
        st.markdown(
            f"<div class='hero'><div>"
            f"<div class='rank'>Post first</div>"
            f"<div class='hook'>{esc(str(r['hook']))}</div>"
            f"<div class='ends'><b>Ends on</b> &nbsp;{esc(str(r['cliffhanger']))}</div>"
            f"<div class='kv'><span class='k'>Cut</span><span class='v'>{cut_text(segs)}</span>"
            f"<span class='k'>Length</span><span class='v'>{r['duration_s']:.0f} seconds</span>"
            f"<span class='k'>Why</span><span class='v'>{esc(str(r['why_it_works']))}</span>"
            f"<span class='k'>Watch out</span><span class='v'>{esc(str(r['weakness']))}</span></div>"
            f"<div class='scores'><div><b>{num(r['hook_score'])}</b>Hook</div><div><b>{num(r['cliff_score'])}</b>Cliffhanger</div>"
            f"<div><b>{num(r['standalone'])}</b>Clear alone</div><div><b>{level(r['spoiler'])}</b>Spoiler risk</div></div>"
            f"</div><div>{clip_html(eps, segs, r['hook_line_id'], r['cliff_line_id'])}"
            f"<div class='legend'><i style='background:var(--accent-soft);margin-left:0'></i>first 3 seconds"
            f"<i style='background:var(--gold-soft)'></i>cliffhanger</div></div></div>",
            unsafe_allow_html=True)

        for _, r in top.iloc[1:].iterrows():
            segs = parse_segments(r["segments"])
            st.markdown(
                f"<div class='row'><div class='n'>{int(r['post_first'])}</div>"
                f"<div><div class='h'>{esc(str(r['hook']))}</div>"
                f"<div class='e'>Ends on: {esc(str(r['cliffhanger']))}</div></div>"
                f"<div class='m'>{cut_text(segs).replace(' + ', '<br>+ ')}<br>{r['duration_s']:.0f}s</div></div>",
                unsafe_allow_html=True)
            with st.expander("Read the clip"):
                st.markdown(f"<div class='t-body' style='margin-bottom:12px'>{esc(str(r['why_it_works']))}</div>"
                            + clip_html(eps, segs, r["hook_line_id"], r["cliff_line_id"]), unsafe_allow_html=True)
    st.write("")
    st.download_button("Download all promos (CSV)", df.to_csv(index=False).encode("utf-8"),
                       f"promos_{Path(run_dir).name}.csv", type="primary")

with t2:
    c1, c2 = st.columns([1, 2])
    min_score = c1.slider("Minimum score", 0.0, 10.0, 0.0, 0.5)
    q = c2.text_input("Search", "", placeholder="Search hooks and cliffhangers")
    view = df[df["score"] >= min_score]
    if q:
        view = view[view["hook"].str.contains(q, case=False, na=False) |
                    view["cliffhanger"].str.contains(q, case=False, na=False)]
    view = view.assign(cut=view["segments"].map(lambda s: cut_text(parse_segments(s))))
    st.dataframe(view[["rank", "cut", "duration_s", "score", "hook", "cliffhanger", "why_it_works"]],
                 hide_index=True, width="stretch",
                 column_config={"rank": "Rank", "cut": "Cut", "duration_s": st.column_config.NumberColumn("Length (s)", format="%d"),
                                "score": st.column_config.NumberColumn("Score", format="%.1f"), "hook": "Hook",
                                "cliffhanger": "Ends on", "why_it_works": "Why it works"})

with t3:
    d = df.assign(episode=df["segments"].map(lambda s: parse_segments(s)[0][0] if parse_segments(s) else None))
    order = sorted(eps) if eps else range(1, int(d["episode"].max()) + 1)
    best = d.groupby("episode")["score"].max().reindex(order).fillna(0)
    best.index = [f"E{int(n):02d}" for n in best.index]
    st.markdown("<div class='t-h2' style='font-size:18px;margin:16px 0 4px'>Best promo per episode</div>"
                "<div class='t-small'>Empty bars: no cut passed the 20–60s, hook and cliffhanger rules.</div>",
                unsafe_allow_html=True)
    st.bar_chart(best.rename("Best score"), height=300, color="#EB0000")

with t4:
    st.markdown(qc_md or "No quality report for this run.")

with t5:
    raw_dir = Path(run_dir) / "raw_llm"
    files = sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []
    if not files:
        st.write("This run made no model calls.")
    else:
        f = st.selectbox("Response", files, format_func=lambda p: p.name)
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        st.markdown(f"<div class='t-small'>{esc(str(d.get('model')))} · {esc(str(d.get('usage')))}</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.text_area("Prompt sent", d.get("prompt", ""), height=420)
        c2.text_area("Raw response", d.get("raw", ""), height=420)
