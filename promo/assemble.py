"""Phase 3: turn hook + cliffhanger candidates into promo cuts. Pure rules, no AI.

Three shapes:
  single     one continuous stretch: hook line -> cliffhanger line, 20-60s
  jump       same episode, 2 segments: short hook beat, then the lead-in to the cliffhanger
  cross      2 segments across episodes N -> N+1: hook beat from N, lead-in to cliffhanger in N+1
Cut points snap to block boundaries so a promo never starts or ends mid-line of dialogue.
"""
from dataclasses import dataclass, field

MIN_LEN, MAX_LEN, TARGET = 20.0, 60.0, 42.0
ASSEMBLY = "v3.1"   # set by run.py --assembly. v3 = original rules; v3.1 = hook fix + prefer continuous clips
CROSS_PRE_PENALTY = {"v3": 1, "v3.1": 3}   # how much a cross-episode montage is handicapped before judging


def is_bare_exclamation(b):
    """'Ahhh!', 'Alexander!', 'Dear God.': loud but meaningless to a stranger."""
    return b.kind == "DIALOGUE" and len(b.text.split()) <= 3


def upgrade_hook(ep, h):
    """v3.1: if the hook is a bare exclamation, start on the ACTION/SOUND line that shows what caused it
    (the slap, the wound, the fire) instead, if one starts within 3s before it."""
    hb = h["block"]
    if ASSEMBLY != "v3.1" or not is_bare_exclamation(hb):
        return h
    near = [b for b in ep.blocks if b.kind in ("ACTION", "SOUND") and hb.start - 3.0 <= b.start <= hb.start + 0.5]
    if not near:
        return h
    best = max(near, key=lambda b: b.start)          # the closest cause before the scream
    return dict(h, block=best, hook_fixed_from=hb.id)


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def dur(self):
        return round(self.end - self.start, 2)


@dataclass
class Promo:
    pid: str
    shape: str
    segments: list
    hook: dict
    cliff: dict
    pre_score: float
    judge: dict = field(default_factory=dict)
    final: float = 0.0

    @property
    def duration(self):
        return round(sum(s.dur for s in self.segments), 2)


def snap_start(ep, t):
    """Move t back to the start of any dialogue line that is playing at t."""
    for b in ep.blocks:
        if b.kind == "DIALOGUE" and b.start < t < b.end:
            t = b.start
    return max(0.0, t)


def snap_end(ep, t):
    """Move t forward to the end of any dialogue line that is playing at t."""
    for b in ep.blocks:
        if b.kind == "DIALOGUE" and b.start < t < b.end:
            t = b.end
    return min(ep.duration, t)


def lead_in_start(ep, end, length, not_before=0.0):
    """Start ~`length` seconds before `end`, snapped forward to a clean line start."""
    x = max(not_before, end - length)
    starts = sorted(b.start for b in ep.blocks if b.kind in ("DIALOGUE", "ACTION") and b.start >= x)
    s = starts[0] if starts else x
    return snap_start(ep, s)


def hook_beat(ep, hb):
    s = snap_start(ep, hb.start)
    e = snap_end(ep, max(hb.end, s + 4.0) + 1.5)   # the hit + the reaction
    return Segment(ep.number, s, min(e, s + 12.0))


def build_for_episode(ep, cands, next_ep=None, next_cands=None, per_ep=4):
    promos = []
    hooks = [upgrade_hook(ep, h) for h in cands["hooks"]]
    cliffs = cands["cliffhangers"]
    for h in hooks:
        hb = h["block"]
        for c in cliffs:
            cb = c["block"]
            if cb.end <= hb.start + 3:
                continue
            pre = h["score"] + c["score"] - (2 if h.get("needs_context") else 0) \
                - (2 if c.get("needs_context") else 0)
            s, e = snap_start(ep, hb.start), snap_end(ep, cb.end)
            if MIN_LEN <= e - s <= MAX_LEN:
                promos.append(Promo("", "single", [Segment(ep.number, s, e)], h, c, pre + 1))
            elif e - s > MAX_LEN:
                seg1 = hook_beat(ep, hb)
                L = TARGET - seg1.dur
                st = lead_in_start(ep, e, L, not_before=seg1.end + 1)
                seg2 = Segment(ep.number, st, e)
                tot = seg1.dur + seg2.dur
                if MIN_LEN <= tot <= MAX_LEN and seg2.dur >= 8:
                    promos.append(Promo("", "jump", [seg1, seg2], h, c, pre))
    # cross-episode: best hook here + best cliffhanger in the next episode
    if next_ep is not None and next_cands and hooks and next_cands["cliffhangers"]:
        h = max(hooks, key=lambda x: x["score"])
        c = max(next_cands["cliffhangers"], key=lambda x: x["score"])
        pen = CROSS_PRE_PENALTY.get(ASSEMBLY, 1)
        seg1 = hook_beat(ep, h["block"])
        e = snap_end(next_ep, c["block"].end)
        seg2 = Segment(next_ep.number, lead_in_start(next_ep, e, TARGET - seg1.dur), e)
        if MIN_LEN <= seg1.dur + seg2.dur <= MAX_LEN and seg2.dur >= 8:
            promos.append(Promo("", "cross", [seg1, seg2], h, c, h["score"] + c["score"] - pen))
    promos = dedupe(sorted(promos, key=lambda p: -p.pre_score))[:per_ep]
    for i, p in enumerate(promos, 1):
        p.pid = f"P{ep.number:02d}-{i}"
    return promos


def overlap(a, b):
    """Fraction of the shorter promo's runtime shared with the other promo."""
    tot = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                tot += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return tot / max(1e-6, min(a.duration, b.duration))


def dedupe(promos, thresh=0.6):
    kept = []
    for p in promos:
        if all(overlap(p, k) < thresh for k in kept):
            kept.append(p)
    return kept


def promo_blocks(eps_by_num, p):
    """Blocks a viewer would see in this promo, in order (used by the judge)."""
    out = []
    for s in p.segments:
        ep = eps_by_num[s.episode]
        out += [b for b in ep.blocks if b.end > s.start + 0.05 and b.start < s.end - 0.05]
    return out
