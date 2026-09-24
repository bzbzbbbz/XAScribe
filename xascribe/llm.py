"""Provider-agnostic client for free, OpenAI-compatible LLM APIs.

XAScribe only needs plain chat completions, so every provider that speaks the
OpenAI `/chat/completions` protocol works.  The default is OpenRouter's free tier
(one free key, no credit card) with open-weight models and automatic fallback.

Configuration (environment variables):
    XASCRIBE_LLM_PROVIDER   openrouter (default) | groq | cerebras | github | nvidia | custom
    XASCRIBE_LLM_MODEL      comma-separated model IDs tried in order (optional)
    XASCRIBE_LLM_BASE_URL   base URL when provider = custom
    <PROVIDER>_API_KEY      e.g. OPENROUTER_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY,
                            GITHUB_TOKEN, NVIDIA_API_KEY; XASCRIBE_LLM_API_KEY for custom
"""
from __future__ import annotations

import contextvars
import os
import re
import time

import requests

# A key pasted by a user in the app applies only to that user's session/thread; it is never stored.
_SESSION_KEY: contextvars.ContextVar = contextvars.ContextVar("xascribe_session_key", default="")


def use_key(key: str) -> None:
    """Use `key` for LLM calls made from the current session (e.g. a Streamlit user) only."""
    _SESSION_KEY.set((key or "").strip())

PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
}

# Free, open-weight models (checked against the OpenRouter catalogue, September 2026).
DEFAULT_MODELS = {
    "openrouter": [
        "z-ai/glm-5.2:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "qwen/qwen3.8-27b:free",
        "google/gemma-4-31b-it:free",
        "openrouter/free",
    ],
    "groq": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    "cerebras": ["gpt-oss-120b", "llama-3.3-70b"],
    "github": ["DeepSeek-V3-0324", "Meta-Llama-3.1-405B-Instruct"],
    "nvidia": ["nvidia/nemotron-3-ultra-550b-a55b", "deepseek-ai/deepseek-v3.1"],
}

_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMUnavailable(RuntimeError):
    """Raised when no provider/key is configured or every model failed."""


def config():
    provider = os.environ.get("XASCRIBE_LLM_PROVIDER", "openrouter").lower()
    if provider == "custom":
        base = os.environ.get("XASCRIBE_LLM_BASE_URL", "")
        key = os.environ.get("XASCRIBE_LLM_API_KEY", "")
        models = []
    else:
        base, key_var = PROVIDERS[provider]
        key = os.environ.get(key_var, "")
        models = DEFAULT_MODELS.get(provider, [])
    if os.environ.get("XASCRIBE_LLM_MODEL"):
        models = [m.strip() for m in os.environ["XASCRIBE_LLM_MODEL"].split(",") if m.strip()]
    key = _SESSION_KEY.get() or key
    return provider, base, key, models


def available() -> bool:
    _, base, key, models = config()
    return bool(base and key and models)


def chat(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1500,
         retries: int | None = None, timeout: int = 300):
    """Return (text, model_id).  Tries each configured model in order.

    Free endpoints are often briefly overloaded (HTTP 429/5xx): each model is retried with
    exponential back-off (XASCRIBE_LLM_RETRIES, default 4) before falling back to the next one.
    Reasoning models spend tokens before answering: an empty, length-truncated answer is retried
    with a larger token budget (up to 16k)."""
    provider, base, key, models = config()
    if not (base and key and models):
        raise LLMUnavailable(f"no API key configured for provider '{provider}'")
    retries = int(os.environ.get("XASCRIBE_LLM_RETRIES", 4)) if retries is None else retries
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers.update({"HTTP-Referer": "https://github.com/xascribe", "X-Title": "XAScribe"})
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    errors = []
    for model in models:
        budget, attempt = max(max_tokens, 1024), 0
        while attempt <= retries:
            try:
                r = requests.post(f"{base}/chat/completions", headers=headers, timeout=timeout,
                                  json={"model": model, "messages": messages,
                                        "temperature": temperature, "max_tokens": budget})
                if r.status_code == 429 or r.status_code >= 500:
                    errors.append(f"{model}: HTTP {r.status_code}")
                    time.sleep(min(120, 10 * 2 ** attempt)); attempt += 1
                    continue
                if r.status_code != 200:
                    errors.append(f"{model}: HTTP {r.status_code} {r.text[:200]}")
                    break
                data = r.json()
                choice = data["choices"][0]
                text = _THINK.sub("", (choice["message"].get("content") or "")).strip()
                if text:
                    return text, data.get("model", model)
                if choice.get("finish_reason") == "length" and budget < 16000:
                    budget = min(16000, budget * 2)      # reasoning consumed the budget; allow more
                    continue
                errors.append(f"{model}: empty response"); attempt += 1
            except (requests.RequestException, KeyError, ValueError) as exc:
                errors.append(f"{model}: {exc}")
                time.sleep(5); attempt += 1
    raise LLMUnavailable("all models failed: " + "; ".join(errors[-6:]))
