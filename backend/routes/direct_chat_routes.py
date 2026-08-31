"""Direct AI chat sessions (ChatGPT-style)."""

from flask import jsonify, request

from backend.middleware.auth import require_auth
from backend.routes import api
from backend.services.groq_service import call_groq
from backend.services.prompt_service import build_general_prompt
from backend.storage.json_store import (
    count,
    delete_by_id,
    delete_where,
    find_all,
    find_one,
    insert,
    now_str,
    update,
)


@api.get("/direct-chats")
@require_auth
def list_direct_chats(user):
    uid = user["id"]
    style = (request.args.get("style") or "").strip().lower()
    if style and style not in ("normal", "radio", "clinic"):
        style = ""

    sessions = find_all(
        "direct_chat_sessions",
        where=lambda s: s.get("user_id") == uid
        and (not style or s.get("style") == style),
        order_by=lambda s: s.get("updated") or s.get("created") or "",
        reverse=True,
        limit=60,
    )
    out = []
    for s in sessions:
        d = dict(s)
        d["message_count"] = count(
            "direct_chat_messages", where=lambda m: m.get("session_id") == s["id"]
        )
        out.append(d)
    return jsonify(out)


@api.post("/direct-chats")
@require_auth
def create_direct_chat(user):
    data = request.get_json(silent=True) or {}
    style = data.get("style", "normal")
    row = insert(
        "direct_chat_sessions",
        {"user_id": user["id"], "title": "New Chat", "style": style, "updated": now_str()},
    )
    return jsonify(
        {
            "id": row["id"],
            "title": "New Chat",
            "style": style,
            "created": row.get("created", ""),
            "updated": row.get("updated", ""),
            "message_count": 0,
        }
    )


@api.get("/direct-chats/<int:sid>")
@require_auth
def get_direct_chat(user, sid):
    session = find_one(
        "direct_chat_sessions",
        lambda s: s.get("id") == sid and s.get("user_id") == user["id"],
    )
    if not session:
        return jsonify({"detail": "Not found"}), 404
    msgs = find_all(
        "direct_chat_messages",
        where=lambda m: m.get("session_id") == sid,
        order_by=lambda m: m.get("created") or "",
    )
    return jsonify({"session": session, "messages": msgs})


@api.post("/direct-chats/<int:sid>/message")
@require_auth
def send_direct_chat_message(user, sid):
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"detail": "Message required"}), 400

    session = find_one(
        "direct_chat_sessions",
        lambda s: s.get("id") == sid and s.get("user_id") == user["id"],
    )
    if not session:
        return jsonify({"detail": "Not found"}), 404

    history = find_all(
        "direct_chat_messages",
        where=lambda m: m.get("session_id") == sid,
        order_by=lambda m: m.get("created") or "",
    )

    system_prompt = build_general_prompt(user.get("role", "patient"), session.get("style"))
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-14:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg})

    try:
        reply = call_groq(messages)
        insert(
            "direct_chat_messages",
            {"session_id": sid, "role": "user", "content": msg},
        )
        insert(
            "direct_chat_messages",
            {"session_id": sid, "role": "assistant", "content": reply},
        )
        if not history:
            title = msg[:55] + ("…" if len(msg) > 55 else "")
            update("direct_chat_sessions", sid, {"title": title, "updated": now_str()})
        else:
            update("direct_chat_sessions", sid, {"updated": now_str()})
        return jsonify({"response": reply})
    except Exception:
        return jsonify({"detail": "Chat service unavailable"}), 503


@api.delete("/direct-chats/<int:sid>")
@require_auth
def delete_direct_chat(user, sid):
    session = find_one(
        "direct_chat_sessions",
        lambda s: s.get("id") == sid and s.get("user_id") == user["id"],
    )
    if not session:
        return jsonify({"ok": True})
    delete_where("direct_chat_messages", lambda m: m.get("session_id") == sid)
    delete_by_id("direct_chat_sessions", sid)
    return jsonify({"ok": True})


@api.patch("/direct-chats/<int:sid>/title")
@require_auth
def rename_direct_chat(user, sid):
    data = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"detail": "Title required"}), 400
    session = find_one(
        "direct_chat_sessions",
        lambda s: s.get("id") == sid and s.get("user_id") == user["id"],
    )
    if not session:
        return jsonify({"detail": "Not found"}), 404
    update("direct_chat_sessions", sid, {"title": title})
    return jsonify({"ok": True})
