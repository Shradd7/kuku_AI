"""Find the best promos in a microdrama.

    python run.py --input "path/to/show_folder" --show "Show name"
    python run.py --input ... --episodes 1-5          # quick test on a few episodes
    python run.py --input ... --provider offline      # no API key, rule-based baseline

Model is set in .env (PROVIDER / MODEL / API key). See README.
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from promo.parse import parse_folder, validate, fmt
from promo.llm import LLM
import promo.assemble as asm_mod
import promo.candidates as cand_mod
from promo.candidates import find_candidates, find_candidates_batch
from promo.assemble import build_for_episode, dedupe
from promo.judge import judge_episode, final_score, post_first
from promo.qc import rule_checks, compare_with_labels


def ep_range(spec, nums):
    if not spec:
        return set(nums)
    a, _, b = spec.partition("-")
    return set(range(int(a), int(b or a) + 1))


def segs_str(p):
    return " + ".join(f"E{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in p.segments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with episode .txt screenplays")
    ap.add_argument("--show", default="", help="show title (defaults to folder name)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--episodes", default="", help="e.g. 1-5")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--labels", default="labels.csv", help="optional hand-picked promos to compare with")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--assembly", default="v3.1", choices=["v3", "v3.1"],
                    help="v3.1 = start bare-exclamation hooks on the action line + prefer continuous clips")
    ap.add_argument("--prompt", default="v4", choices=["v1", "v2", "v3", "v4"],
                    help="v1 original; v2 + rubric, invented examples, series arc for spoilers; "
                         "v3 + exact counts and a mid-episode cliffhanger rule; "
                         "v4 + per-episode output that Gemini is forced to follow (JSON schema)")
    ap.add_argument("--workers", type=int, default=1, help="calls run in parallel (e.g. 4)")
    ap.add_argument("--batch", type=int, default=1,
                    help="episodes per LLM call (e.g. 5): ~15 calls per 50-episode show, fits free daily quotas")
    a = ap.parse_args()

    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    show = a.show or Path(a.input).name
    cand_mod.PROMPT_VERSION = a.prompt
    asm_mod.ASSEMBLY = a.assembly
    eps = parse_folder(a.input)
    issues = validate(eps)
    keep = ep_range(a.episodes, [e.number for e in eps])
    arc = "\n".join(f"E{e.number}: {e.logline}" for e in eps)   # whole series, for spoiler scoring
    eps = [e for e in eps if e.number in keep]
    by_num = {e.number: e for e in eps}
    llm = LLM(provider=a.provider, model=a.model, cache_dir=out / "raw_llm")
    print(f"[1/5] parsed {len(eps)} episodes, {sum(len(e.blocks) for e in eps)} lines; "
          f"parse issues: {len(issues)} | model: {llm.label} | prompt: {a.prompt} | assembly: {a.assembly}")

    cands, stats = {}, {}

    def do_cand(i):
        e = eps[i]
        c, s = find_candidates(e, llm, show, eps[i - 1].logline if i else "")
        print(f"[2/5] E{e.number:02d}: {len(c['hooks'])} hooks, {len(c['cliffhangers'])} cliffhangers "
              f"(bad IDs dropped: {s['bad_id']})", flush=True)
        return e.number, c, s
    if a.batch > 1:
        groups = [list(range(i, min(i + a.batch, len(eps)))) for i in range(0, len(eps), a.batch)]

        def do_group(g):
            res = find_candidates_batch([eps[i] for i in g], llm, show, eps[g[0] - 1].logline if g[0] else "")
            for n, (c, s) in sorted(res.items()):
                print(f"[2/5] E{n:02d}: {len(c['hooks'])} hooks, {len(c['cliffhangers'])} cliffhangers "
                      f"(bad IDs dropped: {s['bad_id']}{', FAILED' if s.get('failed') else ''})", flush=True)
            return res
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            for res in ex.map(do_group, groups):
                for n, (c, s) in res.items():
                    cands[n], stats[n] = c, s
    else:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            for n, c, s in ex.map(do_cand, range(len(eps))):
                cands[n], stats[n] = c, s

    promos_by_ep = {}
    for i, e in enumerate(eps):
        nxt = eps[i + 1] if i + 1 < len(eps) and eps[i + 1].number == e.number + 1 else None
        promos_by_ep[e.number] = build_for_episode(e, cands[e.number], nxt,
                                                   cands.get(nxt.number) if nxt else None)
    print(f"[3/5] assembled {sum(map(len, promos_by_ep.values()))} promos")

    # judge in groups of episodes (2x the candidate batch): fewer calls, and the judge compares
    # many promos side by side, which keeps its scores consistent across the show
    jb = max(1, a.batch * 2) if a.batch > 1 else 1
    ep_keys = list(promos_by_ep)
    jgroups = [[p for n in ep_keys[i:i + jb] for p in promos_by_ep[n]] for i in range(0, len(ep_keys), jb)]
    jgroups = [g for g in jgroups if g]

    def do_judge(g):
        return g, judge_episode(by_num, g, llm, arc)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        judged = list(ex.map(do_judge, jgroups))
    for ps, verdicts in judged:
        for p in ps:
            p.judge = verdicts.get(p.pid) or {"_missing": True}
            p.final = final_score(p)
    allp = dedupe(sorted([p for ps in promos_by_ep.values() for p in ps], key=lambda p: -p.final))
    first = post_first(allp, a.top)
    print(f"[4/5] judged and ranked {len(allp)} promos")

    qc = rule_checks(by_num, allp, stats)
    lab = compare_with_labels(allp, a.labels, set(by_num))

    # ---------------- outputs
    fields = ["rank", "post_first", "promo_id", "shape", "segments", "duration_s", "hook", "cliffhanger",
              "why_it_works", "weakness", "score", "hook_score", "cliff_score", "standalone", "spoiler",
              "hook_line_id", "cliff_line_id"]
    first_ids = {p.pid: i for i, p in enumerate(first, 1)}
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, p in enumerate(allp, 1):
            j = p.judge
            w.writerow({"rank": r, "post_first": first_ids.get(p.pid, ""), "promo_id": p.pid,
                        "shape": p.shape, "segments": segs_str(p), "duration_s": p.duration,
                        "hook": j.get("hook_line") or p.hook.get("moment", ""),
                        "cliffhanger": j.get("cliff_line") or p.cliff.get("open_question", ""),
                        "why_it_works": j.get("why", ""), "weakness": j.get("weakness", ""),
                        "score": p.final, "hook_score": j.get("hook"), "cliff_score": j.get("cliffhanger"),
                        "standalone": j.get("standalone"), "spoiler": j.get("spoiler"),
                        "hook_line_id": p.hook["block"].id, "cliff_line_id": p.cliff["block"].id})

    md = [f"# Promos for: {show}", f"Model: `{llm.label}` | episodes: {len(eps)} | promos: {len(allp)}", "",
          f"## Post these first (top {len(first)}, max one per episode)", ""]
    for i, p in enumerate(first, 1):
        j = p.judge
        md += [f"### {i}. {p.pid} | {p.duration:.0f}s | score {p.final} | {p.shape}",
               f"- **Cut:** {segs_str(p)}",
               f"- **Hook:** {j.get('hook_line') or p.hook.get('moment')}  (`{p.hook['block'].id}`)",
               f"- **Cliffhanger:** {j.get('cliff_line') or p.cliff.get('open_question')}  (`{p.cliff['block'].id}`)",
               f"- **Why it works:** {j.get('why', '')}",
               f"- **Risk:** {j.get('weakness', '')}", ""]
    (out / "promos.md").write_text("\n".join(md), encoding="utf-8")

    (out / "candidates.json").write_text(json.dumps(
        {n: {k: [{kk: vv for kk, vv in c.items() if kk != "block"} | {"start": c["block"].start,
                 "end": c["block"].end} for c in v] for k, v in c_.items()} for n, c_ in cands.items()},
        indent=1, ensure_ascii=False), encoding="utf-8")

    run = {"show": show, "input": str(Path(a.input).resolve()), "model": llm.label, "episodes": len(eps),
           "batch": a.batch, "workers": a.workers, "prompt": a.prompt, "assembly": a.assembly, "seconds": round(time.time() - t0, 1),
           "live_llm_calls": llm.calls, "cached_llm_calls": llm.cached,
           "tokens_in": llm.tokens_in, "tokens_out": llm.tokens_out, "parse_issues": issues,
           "qc": qc, "labels": lab}
    (out / "run_log.json").write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")

    q = qc["pct"]
    rep = [f"# QC report ({llm.label})", "",
           f"- Promos: {qc['promos']} | shapes: {qc['shapes']}",
           f"- Duration 20-60s: {q['dur_ok']}%",
           f"- Clean cuts (no cut inside a dialogue line): {q['clean_cuts']}%",
           f"- Hook line starts in first 3s: {q['hook_in_first_3s']}%",
           f"- Hooks that are a bare exclamation (<=3 words, e.g. 'Ahhh!'): {q['bare_exclamation_hooks']}% "
           f"(fixed to the action line: {qc['hooks_fixed']})",
           f"- Cross-episode montages: {q['cross_share']}%",
           f"- Ends on the cliffhanger line: {q['ends_on_cliff_line']}%",
           f"- Ends at the episode's own ending (within 3s): {q['ends_at_episode_end']}%  "
           "(high = tool mostly re-finds the writer's designed cliffhanger)",
           f"- LLM line IDs returned: {qc['llm_ids_returned']}, invalid & dropped: {qc['llm_bad_ids']}",
           f"- Promos the judge skipped: {qc['judge_missing']}",
           f"- Episodes that failed (API errors; rerun to fill): {qc['failed_episodes'] or 'none'}", ""]
    if qc["problems"]:
        rep += ["## Rule violations"] + [f"- {x}" for x in qc["problems"]] + [""]
    if lab:
        rep += [f"## vs hand-picked promos ({a.labels})",
                f"- Found {lab['found']}/{lab['labels']} (recall {lab['recall']}); "
                f"tool proposed {lab['tool_promos_in_labelled_eps']} promos in those episodes", ""]
        rep += [f"- E{e} {fmt(s)}-{fmt(t)}: best overlap {o} {n}" for e, s, t, o, n in lab["details"]]
    (out / "qc_report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"[5/5] done in {run['seconds']}s -> {out}/promos.csv, promos.md, qc_report.md, raw_llm/ "
          f"| live calls {llm.calls}, tokens in/out {llm.tokens_in}/{llm.tokens_out}")


if __name__ == "__main__":
    main()
