"""Swappable LLM client. One function: client.json(prompt) -> dict.

Provider is chosen by env vars (see .env.example), so swapping models is one line:
  PROVIDER=gemini     GEMINI_API_KEY=...   MODEL=gemini-2.5-flash
  PROVIDER=openai     OPENAI_API_KEY=...   MODEL=...  OPENAI_BASE_URL=https://api.groq.com/openai/v1
                      (any OpenAI-compatible endpoint: Groq, OpenRouter, OpenAI, Kuku's gateway)
  PROVIDER=anthropic  ANTHROPIC_API_KEY=...  MODEL=...
  PROVIDER=offline    no key; rule-based baseline (used for testing and as a comparison)

Every raw response is cached to disk keyed by a hash of (model, prompt). Reruns are free,
and the cache folder doubles as the "raw outputs" for the submission.
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import requests


def load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def extract_json(text: str):
    """Models sometimes wrap JSON in ``` fences or add prose; pull out the object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def normalize(parsed):
    """Models sometimes wrap the object in a list: [ {...} ] or return a bare list of items.
    Always hand callers a dict."""
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        dicts = [x for x in parsed if isinstance(x, dict)]
        if dicts and all(("hooks" in d or "cliffhangers" in d or "clips" in d) for d in dicts):
            merged = {}
            for d in dicts:                    # merge [{hooks..}, {cliffhangers..}] into one object
                for k, v in d.items():
                    merged.setdefault(k, [])
                    merged[k] += v if isinstance(v, list) else [v]
            return merged
        if dicts and all("pid" in d for d in dicts):
            return {"clips": dicts}            # bare list of judged clips
    return {}                                  # unusable -> treated as "found nothing"


class LLM:
    def __init__(self, provider=None, model=None, cache_dir="outputs/raw_llm", sleep=None):
        load_env()
        self.provider = (provider or os.getenv("PROVIDER", "offline")).lower()
        defaults = {"gemini": "gemini-2.5-flash", "openai": "llama-3.3-70b-versatile",
                    "anthropic": "claude-haiku-5-5", "offline": "rules-v1"}
        self.model = model or (defaults["offline"] if self.provider == "offline" else None) \
            or os.getenv("MODEL") or defaults.get(self.provider, "")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        # seconds between live calls, to stay inside free-tier rate limits
        self.sleep = float(sleep if sleep is not None else os.getenv("LLM_SLEEP", "4"))
        self.calls = 0
        self.cached = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.lock = threading.Lock()

    @property
    def offline(self):
        return self.provider == "offline"

    @property
    def label(self):
        return f"{self.provider}:{self.model}"

    def json(self, prompt: str, tag: str = "call", system: str = "", schema: dict = None) -> dict:
        extra = ("\n" + json.dumps(schema, sort_keys=True)) if schema else ""
        key = hashlib.sha256(f"{self.label}\n{system}\n{prompt}{extra}".encode()).hexdigest()[:16]
        path = self.cache / f"{tag}__{key}.json"
        if path.exists():
            with self.lock:
                self.cached += 1
            return normalize(json.loads(path.read_text(encoding="utf-8"))["parsed"])
        last_err = None
        primary = self.model
        fallback = os.getenv("FALLBACK_MODEL", "")
        for attempt in range(5):
            # after 2 overload/timeout failures, switch to the fallback model for this call
            model = fallback if (fallback and attempt >= 2) else primary
            try:
                text, usage = self._call(prompt, system, model, schema)
                usage["model_used"] = model
                parsed = extract_json(text)
                with self.lock:
                    self.calls += 1
                    self.tokens_in += usage.get("in", 0)
                    self.tokens_out += usage.get("out", 0)
                path.write_text(json.dumps({"tag": tag, "model": self.label, "prompt": prompt,
                                            "raw": text, "parsed": parsed, "usage": usage},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
                time.sleep(self.sleep)
                return normalize(parsed)
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout,
                    json.JSONDecodeError, KeyError) as e:
                last_err = e
                wait = 10 * (attempt + 1)
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (400, 401, 403, 404):   # bad key / bad model name: retrying won't help
                    body = e.response.text[:300] if e.response is not None else ""
                    raise RuntimeError(f"LLM call failed with {code} (check API key and MODEL in .env): {body}")
                hint = ""
                if code == 429 and getattr(e, "response", None) is not None:
                    t = e.response.text
                    hint = " [PER-DAY quota]" if "PerDay" in t or "per_day" in t.lower() else " [per-minute limit]"
                    wait = max(wait, 30)
                print(f"  [llm] {tag} ({model}) attempt {attempt + 1} failed ({code or type(e).__name__}){hint}; retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    # ---- providers -------------------------------------------------------
    def _call(self, prompt, system, model, schema=None):
        if self.provider == "gemini":
            return self._gemini(prompt, system, model, schema)
        if self.provider == "openai":
            return self._openai(prompt, system, model)
        if self.provider == "anthropic":
            return self._anthropic(prompt, system, model)
        raise ValueError(f"provider '{self.provider}' has no live API (offline uses rules)")

    def _gemini(self, prompt, system, model, schema=None):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        if model.startswith("gemma"):
            # Gemma (open model on the Gemini API) has no system instruction, JSON mode or thinking config:
            # fold the system text into the prompt and rely on extract_json() to parse the reply.
            text_in = (system + "\n\n" if system else "") + prompt + "\n\nReply with ONE JSON object only, no prose."
            body = {"contents": [{"role": "user", "parts": [{"text": text_in}]}],
                    "generationConfig": {"temperature": 0.2}}
        else:
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if schema:   # v4: Gemini must produce exactly this shape
                body["generationConfig"]["responseSchema"] = schema
        if os.getenv("GEMINI_THINKING") and not model.startswith("gemma"):   # e.g. low: faster, cheaper; picking hooks needs little reasoning
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": os.environ["GEMINI_THINKING"]}
        r = requests.post(url, params={"key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=300)
        r.raise_for_status()
        d = r.json()
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
        u = d.get("usageMetadata", {})
        return text, {"in": u.get("promptTokenCount", 0), "out": u.get("candidatesTokenCount", 0)}

    def _openai(self, prompt, system, model):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = requests.post(f"{base}/chat/completions",
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          json={"model": model, "messages": msgs, "temperature": 0.2,
                                "response_format": {"type": "json_object"}}, timeout=120)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"], {"in": u.get("prompt_tokens", 0),
                                                        "out": u.get("completion_tokens", 0)}

    def _anthropic(self, prompt, system, model):
        # No temperature: current Claude models think adaptively and may reject sampling params.
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                   "anthropic-version": "2023-06-01"},
                          json={"model": model, "max_tokens": 16000,
                                "system": system + "\nReply with a single JSON object only, no prose.",
                                "messages": [{"role": "user", "content": prompt}]}, timeout=300)
        if r.status_code >= 400:
            print("  [llm] anthropic error:", r.text[:300])
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        text = "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")
        return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
