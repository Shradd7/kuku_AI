"""Phase 4: judge each assembled promo the way a new viewer would, then rank.

The judge sees ONLY the lines inside the promo (no logline, no backstory), so its
"makes sense on its own" score is a real cold-viewer test.
"""
from . import assemble as A
from . import candidates as C
from .assemble import promo_blocks
from .candidates import render
from .parse import fmt

JUDGE_SYSTEM = ("You are a tough performance-marketing reviewer for microdrama promos on TikTok/Reels. "
                "You have never seen this show. Judge only what is in the clip.")

JUDGE_PROMPT = """Below are {n} candidate promo clips cut from a microdrama. Each is what a scrolling viewer
would see, in order. You have NO other context about the show, exactly like the viewer.

For each clip score 1-10:
- hook: would the FIRST 3 SECONDS stop a scroll? (shown separately as FIRST_3S)
- cliffhanger: does the LAST moment leave an open question that makes you need the next episode?
- standalone: does it make sense with zero context (who is fighting whom, what is at stake)?
- spoiler: how much does it give away a major twist (10 = ruins a big reveal)?
Also write: hook_line (what the hook is, <=12 words), cliff_line (the open question, <=15 words),
why (one line on why it works), weakness (one line on the main risk).

{extras}Return JSON: {{"clips": [{{"pid": "...", "hook": 0, "cliffhanger": 0, "standalone": 0, "spoiler": 0,
"hook_line": "...", "cliff_line": "...", "why": "...", "weakness": "..."}}]}}

{clips}
"""

W = {"hook": 0.35, "cliffhanger": 0.35, "standalone": 0.20, "spoiler": 0.10}
EARLY_EP_BONUS = 0.4   # viewers who click land on episode 1; early-episode promos match & spoil less
EARLY_EP_CUTOFF = 10


def clip_text(eps_by_num, p):
    blocks = promo_blocks(eps_by_num, p)
    t0 = p.segments[0].start
    first = [b for b in blocks if b.episode == p.segments[0].episode and b.start < t0 + 3.0]
    first3 = " / ".join(f"{b.kind if b.kind != 'DIALOGUE' else b.speaker}: {b.text}" for b in first)
    segs = " + ".join(f"E{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in p.segments)
    body = render([b for b in blocks if b.kind != "CAMERA"] or blocks)
    return f"=== CLIP {p.pid} ({p.duration:.0f}s; {segs}) ===\nFIRST_3S: {first3}\n{body}\n"


def offline_judge(eps_by_num, promos):
    out = {}
    for p in promos:
        blocks = promo_blocks(eps_by_num, p)
        speakers = {b.speaker for b in blocks if b.kind == "DIALOGUE"}
        standalone = max(3, 9 - max(0, len(speakers) - 2))      # more named people = harder to follow
        hk, cf = p.hook.get("score", 5), p.cliff.get("score", 5)
        out[p.pid] = {"hook": hk, "cliffhanger": cf, "standalone": standalone,
                      "spoiler": 6 if p.shape == "cross" else 3,
                      "hook_line": p.hook.get("moment", "")[:80],
                      "cliff_line": p.cliff.get("open_question", p.cliff.get("moment", ""))[:90],
                      "why": "rules baseline: intense opening line + designed episode-end tension",
                      "weakness": "rules cannot judge whether the moment is clear without context"}
    return out


JUDGE_V2 = """
SCORING RUBRIC (use the full range):
- standalone 9-10: a stranger can say who is against whom and what is at stake; 3-5: needs names or
  history the clip never explains; 1-2: confusing.
- spoiler 9-10: gives away the series' central twist or its ending; 4-6: shows a mid-series reveal;
  1-3: early-story conflict only.
""" + "{rubric}" + """
SERIES ARC (use ONLY to score spoiler; score everything else as a viewer who has never seen the show):
{arc}
"""


def judge_episode(eps_by_num, promos, llm, arc=""):
    if not promos:
        return {}
    if llm.offline:
        return offline_judge(eps_by_num, promos)
    ep_tag = promos[0].pid.split("-")[0] + ("-" + promos[-1].pid.split("-")[0] if len({p.pid.split("-")[0] for p in promos}) > 1 else "")
    extras = JUDGE_V2.replace("{rubric}", C.RUBRIC).replace("{arc}", arc or "(not provided)") \
        if C.PROMPT_VERSION in ("v2", "v3", "v4") else ""
    prompt = JUDGE_PROMPT.format(n=len(promos), clips="\n".join(clip_text(eps_by_num, p) for p in promos),
                                 extras=extras)
    try:
        raw = llm.json(prompt, tag=f"judge_{ep_tag}", system=JUDGE_SYSTEM)
    except RuntimeError as e:
        print(f"  [skip] judge {ep_tag}: {str(e)[:160]}", flush=True)
        return {}
    return {str(c.get("pid")): c for c in raw.get("clips", []) if c.get("pid")}


def num(x, default=5.0):
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return default


def final_score(p):
    j = p.judge
    if not j or j.get("_missing"):
        return 0.0   # judge skipped it: rank last, flagged in QC, never silently scored
    s = (W["hook"] * num(j.get("hook")) + W["cliffhanger"] * num(j.get("cliffhanger"))
         + W["standalone"] * num(j.get("standalone")) + W["spoiler"] * (10 - num(j.get("spoiler"), 3)))
    if p.segments[0].episode <= EARLY_EP_CUTOFF:
        s += EARLY_EP_BONUS
    if A.ASSEMBLY == "v3.1":   # the brief: a promo is "usually one continuous stretch"; montages cost edit time
        s -= {"single": 0.0, "jump": 0.2, "cross": 0.6}.get(p.shape, 0.0)
    return round(s, 2)


def post_first(promos, n=10):
    """Top-N with a diversity rule: at most one promo per episode, so the first
    batch tests different story moments instead of five cuts of the same scene."""
    seen, out = set(), []
    for p in promos:
        if p.segments[0].episode in seen:
            continue
        seen.add(p.segments[0].episode)
        out.append(p)
        if len(out) == n:
            break
    return out
