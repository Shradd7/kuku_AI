# Microdrama Promo Finder (Kuku AI Generalist, Problem 1)

Reads a microdrama's timestamped screenplays and returns ranked 20-60s promo cuts:
the episode and start/end of every segment, the hook, the cliffhanger, why it works, plus a quality report.

On a 50-episode show: **15 model calls, about 2.5 minutes, $0 on Gemini's free tier.**

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env            # paste an API key (free Gemini key: aistudio.google.com/apikey)

python run.py --input "path/to/show_folder" --show "Show name" --batch 5            # full show
python run.py --input "path/to/show_folder" --show "Show name" --batch 5 --episodes 1-5   # quick test
python run.py --input "path/to/show_folder" --provider offline                      # no key: rules baseline

streamlit run app.py            # UI: view results, or upload a .zip of a new show and run it
```
`show_folder` = one `.txt` screenplay per episode (the format of the input pack).

### Options
| Flag | Default | What |
|---|---|---|
| `--batch N` | 1 | episodes per LLM call. **5 recommended**: about 15 calls per show, fits free daily quotas |
| `--workers N` | 1 | calls in parallel (paid keys) |
| `--prompt v1/v2/v3` | v3 | prompt version (see below) |
| `--provider / --model` | from `.env` | override the model |
| `--labels file.csv` | labels.csv | hand-picked promos (`episode,start_sec,end_sec,note`) to grade against |

### Swapping the model (one line in `.env`)
`PROVIDER=gemini | openai | anthropic | offline`, `MODEL=...`, plus that provider's key.
`openai` covers any OpenAI-compatible endpoint (Groq, OpenRouter, a company gateway) via `OPENAI_BASE_URL`.
`FALLBACK_MODEL` is used for a call after 2 overload/timeout failures. Tested: Gemini Flash, Flash-Lite, Gemma.

## How it works
```
scripts -> parse (code) -> find hook & cliffhanger lines (LLM, 5 episodes/call)
        -> assemble 20-60s cuts (code) -> judge each clip as a cold viewer (LLM) -> rank (formula) -> QC (code)
```
- **The LLM never writes timestamps.** It points at line IDs (`E07-L042`) and code looks up the times. Invalid IDs are dropped and counted.
- Cuts snap to line boundaries, so no cut starts or ends mid-sentence. Shapes: `single` (continuous), `jump`
  (hook beat + lead-in to the cliffhanger), `cross` (episode N hook + N+1 cliffhanger).
- The judge sees only the clip's own lines, which tests "makes sense on its own". The series arc is given to it
  **only** for spoiler scoring.
- Rank = 0.35 hook + 0.35 cliffhanger + 0.20 standalone + 0.10 (10 - spoiler), +0.4 for episodes 1-10.
  "Post first" = top 10, at most one per episode.
- Robustness: retries with backoff, a fallback model, a failed batch is skipped (a rerun fills it in), list-wrapped JSON
  is normalised, every response is cached (a rerun never pays twice).

### Prompt versions
| | adds | why |
|---|---|---|
| v1 | basic instructions | first version |
| v2 | scoring rubric with anchors, 6 **invented** good/bad examples, series arc for spoilers | consistent scores; examples are not from the show, so they generalise and don't leak the evaluation set |
| v3 | exact counts per episode, "most cliffhangers must be mid-episode", hooks point at the action not "Ah!" | v2 with 10 episodes per call returned 1 moment per episode and 96% of promos were just episode endings; v3 brought that to 21% |

## Outputs (`outputs/`)
| File | What |
|---|---|
| `promos.md` | top 10 to post first, with hook, cliffhanger, why, risk |
| `promos.csv` | every promo, ranked, with segments and all scores |
| `qc_report.md` | rule checks, comparison with hand-picked promos, failed episodes |
| `candidates.json` | every hook / cliffhanger moment the model found |
| `raw_llm/` | every prompt and unedited model response |
| `run_log.json` | model, prompt version, calls, tokens, time |

## Code
`promo/parse.py` parsing · `promo/llm.py` swappable model client · `promo/candidates.py` moment finder, prompts and
rules baseline · `promo/assemble.py` cut rules · `promo/judge.py` judge and ranking · `promo/qc.py` checks · `run.py` CLI ·
`app.py` Streamlit UI

**Not in this repo:** Kuku's input pack and every output that quotes it (confidential; see `.gitignore`).
