"""Health check and model catalog endpoints."""

from flask import jsonify

from backend.config import config
from backend.inference import get_models_info, models_status
from backend.routes import api
from backend.services.llm_service import groq_configured, llm_available, ollama_configured


@api.get("/health")
def health():
    st = models_status()
    return jsonify({
        "status": "ok",
        **st,
        "llm": {
            "available": llm_available(),
            "groq": groq_configured(),
            "ollama": ollama_configured(),
            "ollama_model": config.OLLAMA_MODEL if ollama_configured() else None,
            "fallback": "ollama" if ollama_configured() else None,
        },
    })


@api.get("/models/info")
def models_info():
    """Checkpoint metadata from disk + which model will actually run."""
    return jsonify(get_models_info())
