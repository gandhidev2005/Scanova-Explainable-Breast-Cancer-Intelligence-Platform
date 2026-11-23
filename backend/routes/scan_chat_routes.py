"""Explainable AI chat tied to a scan result."""

from flask import jsonify, request

from backend.middleware.auth import require_auth
from backend.routes import api
from backend.services.groq_service import call_llm
from backend.services.prompt_service import build_scan_prompt, resolve_effective_role
from backend.storage.json_store import find_one, insert


@api.post("/scan/chat")
@require_auth
def scan_chat(user):
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    scan_ctx = data.get("scan_context", {})
    history = data.get("history", [])
    style = (data.get("style") or "").strip().lower()

    if not message:
        return jsonify({"detail": "Message is required"}), 400

    role = user.get("role", "patient")
    effective = role
    if role == "admin":
        if style not in ("normal", "radio", "clinic"):
            style = "normal"
        effective = resolve_effective_role("admin", style)

    system_prompt = build_scan_prompt(role, effective, scan_ctx)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-12:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    scan_id = data.get("scan_id")

    try:
        reply, provider = call_llm(messages)
        if scan_id:
            try:
                row = find_one("scans", lambda s: s.get("id") == int(scan_id))
                if row and (
                    user["role"] == "admin" or row.get("uploaded_by") == user["id"]
                ):
                    insert(
                        "scan_chats",
                        {
                            "scan_id": scan_id,
                            "user_id": user["id"],
                            "role": "user",
                            "content": message,
                        },
                    )
                    insert(
                        "scan_chats",
                        {
                            "scan_id": scan_id,
                            "user_id": user["id"],
                            "role": "assistant",
                            "content": reply,
                        },
                    )
            except Exception as db_err:
                print(f"[Chat DB] Could not save: {db_err}")
        return jsonify({"response": reply, "provider": provider})
    except Exception:
        return jsonify({"detail": "Chat service unavailable"}), 503
