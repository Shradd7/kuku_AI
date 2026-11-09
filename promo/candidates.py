"""Phase 2: find hook moments and cliffhanger moments in each episode.

The LLM reads the episode and points at LINE IDs only. Code then checks every ID exists
(hallucinated IDs are dropped and counted) and looks up the real timestamps.
"""
import re
from .parse import fmt

PROMPT_VERSION = "v3"   # set by run.py --prompt; v1 original, v2 + rubric/examples, v3 + counts/mid-episode rule

# v2 additions -----------------------------------------------------------------------------
RUBRIC = """SCORING RUBRIC (use the full range; most moments are NOT 8+):
Hook score
  9-10: physical or public shock understood in 1 line with zero context (a slap at a wedding, a ring thrown
        in someone's face, "You're fired" in front of everyone). Stops the scroll even on mute.
  6-8 : strong conflict or insult, but needs one extra line to land, or depends on tone of voice.
  3-5 : emotional but generic (crying, staring, a sad look), or a reaction without the action that caused it.
  1-2 : a name, an exclamation or an order with no stakes ("Stop!", "Your Grace.", "Enough!").
Cliffhanger score
  9-10: a threat, reveal or choice is set up and NOT answered: we cut before the answer.
  6-8 : clear tension, but the viewer can roughly guess what happens next.
  3-5 : the tension is resolved within the clip, or ends on a mood instead of a question.
  1-2 : a calm or happy resolution."""

EXAMPLES = """EXAMPLES (invented scenes, NOT from this show, to calibrate you):
GOOD hook (9): At her own engagement party, the bride's mother slaps her and shouts "You were never my daughter."
   -> violent + public + clear stakes in 3 seconds, no backstory needed.
GOOD hook (8): A waiter is mocked by rich guests; he quietly says "I own this hotel."
   -> status reversal in one line.
BAD hook (2): "Stop!"  -> loud but meaningless to a stranger: who? stop what?
BAD hook (3): A woman cries alone at a window.  -> no conflict, no stakes, scroll past.
GOOD cliffhanger (9): "Sign the divorce papers, or your brother goes to prison." She picks up the pen. CUT.
   -> clear choice, answer withheld.
BAD cliffhanger (3): After the fight, the couple hug and smile.  -> resolved; nothing pulls to the next episode."""


V3_RULES = """HARD RULES (v3):
- For EACH episode return EXACTLY 3 hooks and EXACTLY 3 cliffhangers (fewer only if the episode truly has none).
- At most ONE cliffhanger per episode may be in its final 15 seconds. The others MUST be mid-episode turning
  points: a threat, an accusation, a reveal half-made, a choice offered, where the scene keeps going after it.
  Writers end every episode on a cliffhanger already; your value is finding the ones INSIDE the episode.
- If a hook is physical (a slap, a blow, a scream, something thrown), point to the ACTION or SOUND line that
  shows it, not to an exclamation like "Ah!" or a name being shouted."""


V4_FORMAT = """OUTPUT FORMAT (v4): return one entry PER EPISODE, in order, each with exactly 3 hooks and exactly 3
cliffhangers from THAT episode only:
{"episodes": [{"episode": 1, "hooks": [3 items], "cliffhangers": [3 items]}, ...]}
Each hook: {"line_id", "score", "moment", "why", "needs_context"}.
Each cliffhanger: {"line_id", "score", "moment", "open_question", "needs_context"}."""


def batch_schema(n_eps):
    """JSON schema Gemini enforces while generating: n episodes x exactly 3 hooks + 3 cliffhangers."""
    item = {"type": "OBJECT", "properties": {
        "line_id": {"type": "STRING"}, "score": {"type": "NUMBER"}, "moment": {"type": "STRING"},
        "why": {"type": "STRING"}, "open_question": {"type": "STRING"}, "needs_context": {"type": "BOOLEAN"}},
        "required": ["line_id", "score", "moment"]}
    three = {"type": "ARRAY", "items": item, "minItems": 3, "maxItems": 3}
    ep = {"type": "OBJECT", "properties": {"episode": {"type": "INTEGER"}, "hooks": three, "cliffhangers": three},
          "required": ["episode", "hooks", "cliffhangers"]}
    return {"type": "OBJECT", "properties": {"episodes": {"type": "ARRAY", "items": ep,
                                                          "minItems": n_eps, "maxItems": n_eps}},
            "required": ["episodes"]}


def flatten_episodes(raw):
    """v4 replies are {"episodes":[{hooks, cliffhangers}]}; turn into flat hooks/cliffhangers lists."""
    if isinstance(raw, dict) and isinstance(raw.get("episodes"), list):
        flat = {"hooks": [], "cliffhangers": []}
        for e in raw["episodes"]:
            if isinstance(e, dict):
                flat["hooks"] += e.get("hooks") or []
                flat["cliffhangers"] += e.get("cliffhangers") or []
        return flat
    return raw


def v2_block():
    if PROMPT_VERSION == "v4":
        return "\n" + RUBRIC + "\n\n" + EXAMPLES + "\n\n" + V3_RULES + "\n\n" + V4_FORMAT + "\n"
    if PROMPT_VERSION == "v2":
        return "\n" + RUBRIC + "\n\n" + EXAMPLES + "\n"
    if PROMPT_VERSION == "v3":
        return "\n" + RUBRIC + "\n\n" + EXAMPLES + "\n\n" + V3_RULES + "\n"
    return ""


SYSTEM = ("You are a senior promo editor for vertical microdrama apps (ReelShort, DramaBox). "
          "You pick moments from scripts that make a scrolling viewer stop, and moments that "
          "leave them needing the next episode. You are strict and only pick genuinely strong moments.")

CANDIDATE_PROMPT = """Below is one episode of the microdrama "{show}" as timestamped script lines.
Each line starts with its LINE ID. Context for you (a new viewer will NOT have this):
- Story so far: {prev}
- This episode: {logline}

Find:
1. HOOKS (up to 4): a single line or moment that would stop a scrolling viewer in the first 3 seconds
   of a promo with ZERO context: a slap, a violent act, a public humiliation, a shocking reveal,
   a provocative line. It must be understandable without knowing the characters.
2. CLIFFHANGERS (up to 4): a moment to END a promo on. It must leave a clear unresolved question
   (a threat not carried out, a reveal half-made, a decision not yet taken). Not a moment that is
   resolved a few seconds later.

Rules:
- Use only LINE IDs that appear below. Never invent IDs or timestamps.
- Score 1-10 honestly; 8+ means you would bet money it performs on TikTok.
- "needs_context": true if the moment only lands for someone who knows the backstory.

{extras}Return JSON exactly like:
{{"hooks": [{{"line_id": "E01-L012", "score": 8, "moment": "short description", "why": "one line", "needs_context": false}}],
  "cliffhangers": [{{"line_id": "E01-L090", "score": 7, "moment": "short description", "open_question": "what the viewer wants to know", "needs_context": false}}]}}

SCRIPT:
{script}
"""


def render(blocks, include_camera=True) -> str:
    out = []
    for b in blocks:
        if b.kind == "CAMERA" and not include_camera:
            continue
        who = f"{b.speaker} ({b.emotion})" if b.kind == "DIALOGUE" else b.kind
        out.append(f"{b.id} [{fmt(b.start)}-{fmt(b.end)}] {who}: {b.text}")
    return "\n".join(out)


# ---------------------------------------------------------------- offline baseline
HOOK_WORDS = {
    "slap": 4, "strikes": 3, "punch": 3, "shove": 3, "grabs": 2, "throws": 2, "kneel": 2, "drags": 3,
    "blood": 3, "dagger": 3, "sword": 3, "knife": 3, "gun": 3, "pistol": 3, "choke": 3, "kiss": 2,
    "reveal": 3, "gasps": 2, "gasp": 2, "shatter": 2, "crash": 2, "tears": 1, "screams": 3, "scream": 3,
    "humiliat": 3, "abandon": 2, "betray": 3, "divorce": 3, "pregnant": 3, "dead": 2, "kill": 3,
    "duke": 2, "liar": 2, "whore": 3, "beggar": 2, "cripple": 3, "arrest": 3, "poison": 3, "fire": 2,
}
HOT_EMOTIONS = ["furious", "shocked", "screaming", "venomous", "cruel", "mocking", "enraged",
                "horrified", "desperate", "menacing", "icy", "contempt", "sneer", "explosive", "livid"]
CLIFF_WORDS = {"cliffhanger": 5, "ominous": 2, "freeze": 2, "frozen": 2, "cut to black": 3,
               "who": 1, "what?": 2, "unless": 2, "or else": 3, "you will": 2, "until": 1,
               "reveals": 2, "door bursts": 3, "stops": 1, "?": 1, "sting": 2, "swell": 1}


def _score(text, table):
    t = text.lower()
    return sum(w for k, w in table.items() if k in t)


def offline_candidates(ep, k=4):
    hooks, cliffs = [], []
    n = len(ep.blocks)
    for i, b in enumerate(ep.blocks):
        txt = b.text + " " + b.emotion
        h = _score(txt, HOOK_WORDS) + 2 * sum(e in b.emotion.lower() for e in HOT_EMOTIONS)
        if b.kind == "DIALOGUE":
            h += 1 + (2 if "!" in b.text else 0)
        if h >= 4:
            hooks.append({"line_id": b.id, "score": min(10, 3 + h), "moment": b.text[:90],
                          "why": "rule match: high-intensity words/emotion", "needs_context": False})
        c = _score(txt, CLIFF_WORDS)
        c += 4 * (i >= n - 4)  # writers put the designed cliffhanger at the episode end
        if c >= 4:
            cliffs.append({"line_id": b.id, "score": min(10, 2 + c), "moment": b.text[:90],
                           "open_question": "rule match: tension words / episode end", "needs_context": False})
    hooks = sorted(hooks, key=lambda x: -x["score"])[:k]
    cliffs = sorted(cliffs, key=lambda x: -x["score"])[:k]
    return {"hooks": hooks, "cliffhangers": cliffs}


# ---------------------------------------------------------------- main entry
def find_candidates(ep, llm, show, prev_logline=""):
    if llm.offline:
        raw = offline_candidates(ep)
    else:
        prompt = CANDIDATE_PROMPT.format(show=show, prev=prev_logline or "(first episode)",
                                         logline=ep.logline, script=render(ep.blocks), extras=v2_block())
        try:
            raw = llm.json(prompt, tag=f"cand_E{ep.number:02d}", system=SYSTEM)
        except RuntimeError as e:          # one failed episode must not kill the run; rerun fills it in
            print(f"  [skip] E{ep.number:02d}: {str(e)[:160]}", flush=True)
            return {"hooks": [], "cliffhangers": []}, {"returned": 0, "bad_id": 0, "failed": 1}
    ids = {b.id: b for b in ep.blocks}
    stats = {"returned": 0, "bad_id": 0, "failed": 0}
    out = {"hooks": [], "cliffhangers": []}
    for kind in ("hooks", "cliffhangers"):
        for c in raw.get(kind, []) or []:
            stats["returned"] += 1
            lid = str(c.get("line_id", "")).strip()
            if lid not in ids:          # hallucinated or malformed ID -> drop, count
                stats["bad_id"] += 1
                continue
            try:
                c["score"] = float(c.get("score", 0))
            except (TypeError, ValueError):
                c["score"] = 0.0
            c["block"] = ids[lid]
            out[kind].append(c)
    return out, stats


BATCH_PROMPT = """Below are {n} consecutive episodes of the microdrama "{show}" as timestamped script lines.
Each line starts with its LINE ID (E<episode>-L<line>). Context for you (a new viewer will NOT have this):
- Story before these episodes: {prev}
{loglines}

For EACH episode separately, find:
1. HOOKS (up to 4 per episode): a single line or moment that would stop a scrolling viewer in the first 3 seconds
   of a promo with ZERO context: a slap, a violent act, a public humiliation, a shocking reveal,
   a provocative line. It must be understandable without knowing the characters.
2. CLIFFHANGERS (up to 4 per episode): a moment to END a promo on. It must leave a clear unresolved question
   (a threat not carried out, a reveal half-made, a decision not yet taken). Not a moment that is
   resolved a few seconds later.

Rules:
- Use only LINE IDs that appear below. Never invent IDs or timestamps.
- Every episode below must get its own hooks and cliffhangers.
- Score 1-10 honestly; 8+ means you would bet money it performs on TikTok.
- "needs_context": true if the moment only lands for someone who knows the backstory.

{extras}Return ONE JSON object exactly like:
{{"hooks": [{{"line_id": "E01-L012", "score": 8, "moment": "short description", "why": "one line", "needs_context": false}}],
  "cliffhangers": [{{"line_id": "E01-L090", "score": 7, "moment": "short description", "open_question": "what the viewer wants to know", "needs_context": false}}]}}

SCRIPTS:
{scripts}
"""


def find_candidates_batch(batch, llm, show, prev_logline=""):
    """Several episodes in ONE call (free tiers allow ~20 calls/day/model). Answers are routed back to
    each episode by the E<nn> prefix of the line ID, and validated against that episode's lines."""
    if llm.offline:
        return {e.number: find_candidates(e, llm, show)[0:2] for e in batch}
    prompt = BATCH_PROMPT.format(
        n=len(batch), show=show, prev=prev_logline or "(first episode)", extras=v2_block(),
        loglines="\n".join(f"- Episode {e.number}: {e.logline}" for e in batch),
        scripts="\n\n".join(f"=== EPISODE {e.number} ===\n{render(e.blocks)}" for e in batch))
    tag = f"candB_E{batch[0].number:02d}-E{batch[-1].number:02d}"
    out = {e.number: ({"hooks": [], "cliffhangers": []}, {"returned": 0, "bad_id": 0, "failed": 0}) for e in batch}
    if PROMPT_VERSION == "v4":   # the generic example block below asks for a flat list; v4 overrides it
        prompt = prompt.replace("Return ONE JSON object exactly like:", "Ignore this older flat example; use the OUTPUT FORMAT (v4) above:")
    try:
        schema = batch_schema(len(batch)) if PROMPT_VERSION == "v4" and llm.provider == "gemini" \
            and not llm.model.startswith("gemma") else None
        raw = flatten_episodes(llm.json(prompt, tag=tag, system=SYSTEM, schema=schema))
    except RuntimeError as err:
        print(f"  [skip] {tag}: {str(err)[:160]}", flush=True)
        for e in batch:
            out[e.number][1]["failed"] = 1
        return out
    ids = {b.id: b for e in batch for b in e.blocks}
    for kind in ("hooks", "cliffhangers"):
        for c in raw.get(kind, []) or []:
            lid = str(c.get("line_id", "")).strip()
            m = re.match(r"E(\d+)-L\d+$", lid)
            n = int(m.group(1)) if m else None
            if n in out:
                out[n][1]["returned"] += 1
            if lid not in ids:
                if n in out:
                    out[n][1]["bad_id"] += 1
                else:   # unroutable id: charge it to the first episode so it is still counted
                    out[batch[0].number][1]["returned"] += 1
                    out[batch[0].number][1]["bad_id"] += 1
                continue
            try:
                c["score"] = float(c.get("score", 0))
            except (TypeError, ValueError):
                c["score"] = 0.0
            c["block"] = ids[lid]
            out[ids[lid].episode][0][kind].append(c)
    for n, (c, st) in out.items():   # keep at most 4 of each per episode
        for kind in ("hooks", "cliffhangers"):
            c[kind] = sorted(c[kind], key=lambda x: -x["score"])[:4]
    return out
