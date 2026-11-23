"""
Unified LLM caller: Groq first, then local Ollama if the API key fails
or Groq is unreachable.

Both backends receive the *same* messages (system + history + user) and
matching generation settings so Ollama responses follow the same Scanova
prompts / style as Groq.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import requests

from backend.config import config

log = logging.getLogger(__name__)

# Shared generation knobs — keep identical across providers
_MAX_TOKENS = 1024
_TEMPERATURE = 0.6
_TIMEOUT_GROQ = 45
_TIMEOUT_OLLAMA = 120


def groq_configured() -> bool:
    key = config.GROQ_API_KEY
    return bool(
        key
        and not key.startswith("your-")
        and config.GROQ_API_URL
        and config.GROQ_MODEL
    )


def ollama_configured() -> bool:
    return bool(
        config.OLLAMA_ENABLED
        and config.OLLAMA_BASE_URL
        and config.OLLAMA_MODEL
    )


def llm_available() -> bool:
    return groq_configured() or ollama_configured()


def _extract_content(data: dict) -> str:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        content = (message.get("reasoning") or "").strip()
    return content


def _ollama_messages(messages: list) -> list:
    """
    Same conversation as Groq, plus a short style lock so local models
    stick to Scanova's system-prompt format (structure, tone, disclaimers).
    """
    msgs = deepcopy(messages)
    style = (
        "\n\n[Output contract — follow exactly]\n"
        "- Obey the system instructions above completely.\n"
        "- Match the same tone, structure, and clinical caution as a cloud LLM "
        "answering under those instructions.\n"
        "- Prefer the same headings, bullet layout, and markdown tables when useful.\n"
        "- Do not mention that you are a local model, Ollama, or a fallback.\n"
        "- Do not invent scan metrics not present in the system context."
    )
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = (msgs[0].get("content") or "") + style
    else:
        msgs.insert(0, {"role": "system", "content": style.strip()})
    return msgs


def _call_groq(messages: list) -> str:
    if not groq_configured():
        raise RuntimeError("Groq not configured")

    resp = requests.post(
        config.GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.GROQ_MODEL,
            "messages": messages,
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
        },
        timeout=_TIMEOUT_GROQ,
    )
    if not resp.ok:
        detail = ""
        try:
            detail = (resp.json().get("error") or {}).get("message") or ""
        except Exception:
            detail = ""
        log.error(
            "Groq API HTTP %s model=%s detail=%s",
            resp.status_code,
            config.GROQ_MODEL,
            (detail or resp.reason)[:200],
        )
        raise RuntimeError(f"Groq HTTP {resp.status_code}")

    content = _extract_content(resp.json())
    if not content:
        raise RuntimeError("Groq returned an empty response")
    return content


def _call_ollama(messages: list) -> str:
    if not ollama_configured():
        raise RuntimeError("Ollama not configured")

    base = config.OLLAMA_BASE_URL.rstrip("/")
    # Prefer OpenAI-compatible endpoint (same payload shape as Groq)
    url = f"{base}/v1/chat/completions"

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "model": config.OLLAMA_MODEL,
            "messages": _ollama_messages(messages),
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "stream": False,
        },
        timeout=_TIMEOUT_OLLAMA,
    )
    if not resp.ok:
        # Fallback to native /api/chat if v1 is unavailable
        if resp.status_code in (404, 405):
            return _call_ollama_native(messages)
        log.error(
            "Ollama HTTP %s model=%s body=%s",
            resp.status_code,
            config.OLLAMA_MODEL,
            (resp.text or "")[:200],
        )
        raise RuntimeError(f"Ollama HTTP {resp.status_code}")

    content = _extract_content(resp.json())
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content


def _call_ollama_native(messages: list) -> str:
    """Native Ollama /api/chat when OpenAI-compat shim is missing."""
    base = config.OLLAMA_BASE_URL.rstrip("/")
    resp = requests.post(
        f"{base}/api/chat",
        headers={"Content-Type": "application/json"},
        json={
            "model": config.OLLAMA_MODEL,
            "messages": _ollama_messages(messages),
            "stream": False,
            "options": {
                "temperature": _TEMPERATURE,
                "num_predict": _MAX_TOKENS,
            },
        },
        timeout=_TIMEOUT_OLLAMA,
    )
    if not resp.ok:
        log.error(
            "Ollama native HTTP %s model=%s",
            resp.status_code,
            config.OLLAMA_MODEL,
        )
        raise RuntimeError(f"Ollama HTTP {resp.status_code}")

    data = resp.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content


def call_llm(messages: list) -> tuple[str, str]:
    """
    Returns (reply_text, provider) where provider is 'groq' | 'ollama'.

    Order: Groq → Ollama (only if Groq missing/fails and Ollama is enabled).
    """
    if not llm_available():
        raise RuntimeError("AI chat is not configured")

    groq_err: Exception | None = None
    if groq_configured():
        try:
            return _call_groq(messages), "groq"
        except Exception as exc:
            groq_err = exc
            log.warning("Groq failed (%s) — trying Ollama fallback", type(exc).__name__)

    if ollama_configured():
        try:
            return _call_ollama(messages), "ollama"
        except Exception as exc:
            log.error("Ollama fallback failed: %s", type(exc).__name__)
            if groq_err:
                raise RuntimeError("AI chat service unavailable") from exc
            raise RuntimeError("AI chat service unavailable") from exc

    raise RuntimeError("AI chat service unavailable")


def call_groq(messages: list) -> str:
    """Back-compat entry point used by chat routes — includes Ollama fallback."""
    text, _provider = call_llm(messages)
    return text
