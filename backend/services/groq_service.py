"""Groq LLM integration — API key read from .env only, never logged or returned."""

import logging

import requests

from backend.config import config

log = logging.getLogger(__name__)


def groq_configured() -> bool:
    key = config.GROQ_API_KEY
    return bool(key) and not key.startswith("your-")


def call_groq(messages: list) -> str:
    if not groq_configured():
        raise RuntimeError("AI chat is not configured")

    try:
        resp = requests.post(
            config.GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.GROQ_MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.6,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as exc:
        log.error("Groq API request failed: %s", type(exc).__name__)
        raise RuntimeError("AI chat service unavailable") from exc
