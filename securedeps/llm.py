"""Thin clients for two LLM providers. The rest of the app only calls  llm.complete(system, user)
and gets text back, so the provider can be swapped without touching anything else.

  Gemini     set GEMINI_API_KEY      (free tier available)   optional GEMINI_MODEL
  Anthropic  set ANTHROPIC_API_KEY   (paid API)              optional ANTHROPIC_MODEL

If both keys are set, LLM_PROVIDER=gemini|anthropic chooses. If neither is set, get_llm()
returns None and the app falls back to its deterministic (non-AI) explanation.
"""
import os
import time
from typing import Optional

import requests


class LLMError(Exception):
    pass


def _retry_after(r) -> int:
    """Seconds the server asked us to wait (Retry-After header), capped at one minute; 0 if absent."""
    try:
        return min(int((getattr(r, "headers", None) or {}).get("Retry-After", 0)), 60)
    except (TypeError, ValueError):
        return 0


def _post(session, url: str, headers: dict, body: dict) -> dict:
    r = None
    for attempt in range(3):          # retry timeouts and 5xx ("model overloaded" is common on free tiers)
        last = attempt == 2
        try:
            r = session.post(url, headers=headers, json=body, timeout=120)
        except requests.Timeout:
            if last:
                raise LLMError("the LLM took too long to answer (timed out 3 times). "
                               "Try a faster model, e.g. GEMINI_MODEL=gemini-3.5-flash-lite")
            time.sleep(2 * (attempt + 1))
            continue
        except requests.RequestException as e:
            raise LLMError(f"network error talking to the LLM ({type(e).__name__})")
        if r.status_code >= 500 and not last:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 and not last:     # free-tier per-minute limit: wait, then try again
            time.sleep(_retry_after(r) or 20 * (attempt + 1))
            continue
        break
    if r.status_code == 429:
        raise LLMError("LLM rate limit reached (common on free tiers). Wait a minute and retry.")
    if r.status_code in (401, 403):
        raise LLMError("the LLM rejected the API key. Check the key and that the API is enabled.")
    if r.status_code == 404:
        raise LLMError("LLM model name not found. Set GEMINI_MODEL / ANTHROPIC_MODEL to a current model.")
    if not r.ok:
        try:
            detail = str(r.json().get("error", {}).get("message", ""))[:160]
        except Exception:
            detail = ""
        hint = (" (this usually means the API key is invalid or deleted: export GEMINI_API_KEY again)"
                if "api key" in detail.lower() else "")
        raise LLMError(f"LLM returned HTTP {r.status_code}: {detail}{hint}" if detail
                       else f"LLM returned HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError:
        raise LLMError("LLM returned something that is not JSON")


class GeminiLLM:
    def __init__(self, api_key: str, model: Optional[str] = None, session=None):
        self.key = api_key
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.session = session or requests.Session()

    def complete(self, system: str, user: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = _post(self.session, url, {"x-goog-api-key": self.key, "Content-Type": "application/json"},
                     {"systemInstruction": {"parts": [{"text": system}]},
                      "contents": [{"role": "user", "parts": [{"text": user}]}],
                      "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}})
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            reason = (data.get("promptFeedback") or {}).get("blockReason") or "no answer returned"
            raise LLMError(f"Gemini gave no usable answer ({reason})")


class AnthropicLLM:
    def __init__(self, api_key: str, model: Optional[str] = None, session=None):
        self.key = api_key
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.session = session or requests.Session()

    def complete(self, system: str, user: str) -> str:
        data = _post(self.session, "https://api.anthropic.com/v1/messages",
                     {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                      "content-type": "application/json"},
                     {"model": self.model, "max_tokens": 1500, "temperature": 0.1, "system": system,
                      "messages": [{"role": "user", "content": user}]})
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError):
            raise LLMError("Anthropic gave no usable answer")


def get_llm(env=None, session=None):
    env = os.environ if env is None else env
    gem, ant = env.get("GEMINI_API_KEY"), env.get("ANTHROPIC_API_KEY")
    choice = (env.get("LLM_PROVIDER") or "").lower()
    if choice == "anthropic" and ant or (not choice and ant and not gem):
        return AnthropicLLM(ant, session=session)
    if gem and choice in ("", "gemini"):
        return GeminiLLM(gem, session=session)
    return None
