"""Phase 1: parse screenplay .txt files into a flat table of timestamped blocks.

Pure code, no AI. Every block gets a stable ID (E07-L042) that the LLM refers to,
so the model never has to write timestamps itself.
"""
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

TS = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"
BLOCK_HEAD = re.compile(r"^\*\*(?P<label>[^*]+?):\*\*\s*$")
BLOCK_TIME = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)\s*$")
SCENE_HEAD = re.compile(r"^SCENE\s+(\d+)")


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Block:
    id: str
    episode: int
    scene: int
    idx: int
    kind: str          # DIALOGUE | ACTION | CAMERA | SOUND
    speaker: str       # for dialogue
    emotion: str       # parenthetical on dialogue, e.g. "Cold, Commanding"
    start: float
    end: float
    text: str


@dataclass
class Episode:
    number: int
    file: str
    logline: str = ""
    continuity: str = ""
    duration: float = 0.0
    blocks: list = field(default_factory=list)


def parse_file(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ep = Episode(number=0, file=path.name)
    scene, i, idx = 0, 0, 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("EPISODE NUMBER:"):
            m = re.search(r"(\d+)", ln)
            ep.number = int(m.group(1)) if m else 0
        elif ln.startswith("EPISODE LOGLINE:"):
            ep.logline = ln.split(":", 1)[1].strip()
        elif ln.startswith("CONTINUITY NOTES:"):
            ep.continuity = ln.split(":", 1)[1].strip()
        elif SCENE_HEAD.match(ln):
            scene = int(SCENE_HEAD.match(ln).group(1))
        else:
            h = BLOCK_HEAD.match(ln)
            if h and i + 1 < len(lines) and BLOCK_TIME.match(lines[i + 1].strip()):
                label = h.group("label").strip()
                t = BLOCK_TIME.match(lines[i + 1].strip())
                start, end = to_sec(t.group(1), t.group(2)), to_sec(t.group(3), t.group(4))
                # body = following lines until blank line / next block
                body, j = [], i + 2
                while j < len(lines) and lines[j].strip() and not BLOCK_HEAD.match(lines[j].strip()) \
                        and not lines[j].startswith("---"):
                    body.append(lines[j].strip())
                    j += 1
                text = " ".join(body).strip()
                up = label.upper()
                if up == "CAMERA":
                    kind, spk, emo = "CAMERA", "", ""
                elif up == "ACTION":
                    kind, spk, emo = "ACTION", "", ""
                elif up.startswith("SOUND"):
                    kind, spk, emo = "SOUND", "", ""
                else:
                    kind = "DIALOGUE"
                    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
                    spk, emo = (m.group(1), m.group(2)) if m else (label, "")
                    text = text.strip('"').strip("“”")
                idx += 1
                ep.blocks.append(Block("", ep.number, scene, idx, kind, spk.strip(), emo.strip(),
                                       start, end, text))
                i = j
                continue
        i += 1
    if not ep.number:  # fall back to file name
        m = re.search(r"episode-(\d+)", path.name)
        ep.number = int(m.group(1)) if m else 0
    for b in ep.blocks:
        b.episode = ep.number
        b.id = f"E{ep.number:02d}-L{b.idx:03d}"
    ep.duration = max((b.end for b in ep.blocks), default=0.0)
    return ep


def parse_folder(folder) -> list:
    files = sorted(Path(folder).rglob("*.txt"))
    eps = [parse_file(f) for f in files]
    eps = [e for e in eps if e.blocks]
    return sorted(eps, key=lambda e: e.number)


def validate(eps) -> list:
    """Return a list of human-readable problems found in the parse."""
    issues = []
    for e in eps:
        for b in e.blocks:
            if b.end < b.start:
                issues.append(f"{b.id}: end before start")
            if not b.text:
                issues.append(f"{b.id}: empty text")
        if e.duration < 20:
            issues.append(f"Episode {e.number}: very short ({e.duration:.1f}s)")
    nums = [e.number for e in eps]
    if len(nums) != len(set(nums)):
        issues.append("duplicate episode numbers")
    return issues


def blocks_as_rows(eps):
    return [asdict(b) for e in eps for b in e.blocks]
