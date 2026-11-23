"""Groq LLM integration — delegates to llm_service (Groq → Ollama fallback).

API key is read from .env only and never logged or returned to clients.
"""

from backend.services.llm_service import (  # noqa: F401
    call_groq,
    call_llm,
    groq_configured,
    llm_available,
    ollama_configured,
)

__all__ = [
    "call_groq",
    "call_llm",
    "groq_configured",
    "llm_available",
    "ollama_configured",
]
