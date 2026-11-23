"""General AI chat (no scan context)."""

from flask import jsonify, request

from backend.middleware.auth import require_auth
from backend.routes import api
from backend.services.groq_service import call_llm
from backend.services.prompt_service import build_general_prompt


@api.post("/chat/general")
@require_auth
def general_chat(user):
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])
    style = data.get("style", None)

    if not message:
        return jsonify({"detail": "Message is required"}), 400

    system_prompt = build_general_prompt(user.get("role", "patient"), style)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-14:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        reply, provider = call_llm(messages)
        return jsonify({"response": reply, "provider": provider})
    except Exception:
        return jsonify({"detail": "Chat service unavailable"}), 503
