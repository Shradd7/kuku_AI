"""Phase 5: automatic quality checks + optional comparison with hand-picked promos."""
import csv
from pathlib import Path


def mid_dialogue(ep, t, edge):
    for b in ep.blocks:
        if b.kind == "DIALOGUE" and b.start + 0.05 < t < b.end - 0.05:
            return True
    return False


def rule_checks(eps_by_num, promos, cand_stats):
    n = len(promos) or 1
    res = {"promos": len(promos), "dur_ok": 0, "clean_cuts": 0, "hook_in_first_3s": 0,
           "ends_on_cliff_line": 0, "ends_at_episode_end": 0, "judge_missing": 0,
           "shapes": {}, "problems": []}
    for p in promos:
        res["shapes"][p.shape] = res["shapes"].get(p.shape, 0) + 1
        if 20 <= p.duration <= 60:
            res["dur_ok"] += 1
        else:
            res["problems"].append(f"{p.pid}: duration {p.duration}s")
        clean = True
        for s in p.segments:
            ep = eps_by_num[s.episode]
            if mid_dialogue(ep, s.start, "start") or mid_dialogue(ep, s.end, "end"):
                clean = False
                res["problems"].append(f"{p.pid}: cut inside a dialogue line (E{s.episode})")
            if s.start < 0 or s.end > ep.duration + 0.01:
                res["problems"].append(f"{p.pid}: segment outside episode bounds")
        res["clean_cuts"] += clean
        hb, cb = p.hook["block"], p.cliff["block"]
        if hb.start - p.segments[0].start <= 3.0:
            res["hook_in_first_3s"] += 1
        last = p.segments[-1]
        if abs(last.end - cb.end) <= 2.0:
            res["ends_on_cliff_line"] += 1
        if eps_by_num[last.episode].duration - last.end <= 3.0:
            res["ends_at_episode_end"] += 1
        if not p.judge or p.judge.get("_missing"):
            res["judge_missing"] += 1
    ret = sum(s["returned"] for s in cand_stats.values())
    bad = sum(s["bad_id"] for s in cand_stats.values())
    res["llm_ids_returned"], res["llm_bad_ids"] = ret, bad
    res["failed_episodes"] = sorted(n for n, s in cand_stats.items() if s.get("failed"))
    res["pct"] = {k: round(100 * res[k] / n, 1) for k in
                  ("dur_ok", "clean_cuts", "hook_in_first_3s", "ends_on_cliff_line", "ends_at_episode_end")}
    return res


def compare_with_labels(promos, labels_csv, run_eps=None):
    """labels.csv columns: episode,start_sec,end_sec,note  (promos an editor picked by hand).
    A label counts as found if any tool promo overlaps at least 50% of it."""
    p = Path(labels_csv)
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if run_eps is not None:                      # only grade labels for episodes that were run
        rows = [r for r in rows if int(r["episode"]) in run_eps]
    if not rows:
        return None
    labelled_eps = {int(r["episode"]) for r in rows}
    found, details = 0, []
    for r in rows:
        e, s, t = int(r["episode"]), float(r["start_sec"]), float(r["end_sec"])
        best = 0.0
        for pr in promos:
            for seg in pr.segments:
                if seg.episode == e:
                    best = max(best, max(0.0, min(t, seg.end) - max(s, seg.start)) / max(1e-6, t - s))
        found += best >= 0.5
        details.append((e, s, t, round(best, 2), r.get("note", "")))
    tool_in_labelled = [pr for pr in promos if pr.segments[0].episode in labelled_eps]
    return {"labels": len(rows), "found": found, "recall": round(found / max(1, len(rows)), 2),
            "tool_promos_in_labelled_eps": len(tool_in_labelled), "details": details}
