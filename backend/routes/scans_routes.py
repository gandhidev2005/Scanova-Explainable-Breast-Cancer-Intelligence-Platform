"""Scan CRUD and scan chat history."""

from flask import jsonify, request

from backend.middleware.auth import require_auth
from backend.routes import api
from backend.storage.json_store import count, delete_by_id, delete_where, find_all, find_one, update


@api.get("/scans")
@require_auth
def get_scans(user):
    rows = find_all(
        "scans",
        where=lambda s: s.get("uploaded_by") == user["id"],
        order_by=lambda s: s.get("created") or "",
        reverse=True,
    )
    result = []
    for r in rows:
        d = dict(r)
        d.pop("heatmap", None)
        if d.get("uploaded_by") != user["id"]:
            d["label"] = None
        d["chat_count"] = count(
            "scan_chats", where=lambda c: c.get("scan_id") == d["id"]
        )
        result.append(d)
    return jsonify(result)


@api.get("/scans/<int:scan_id>")
@require_auth
def get_scan(user, scan_id):
    row = find_one("scans", lambda s: s.get("id") == scan_id)
    if not row:
        return jsonify({"detail": "Scan not found"}), 404
    d = dict(row)
    if user["role"] != "admin" and d.get("uploaded_by") != user["id"]:
        return jsonify({"detail": "Access denied"}), 403
    if d.get("uploaded_by") != user["id"]:
        d["label"] = None
    return jsonify(d)


@api.patch("/scans/<int:scan_id>/label")
@require_auth
def label_scan(user, scan_id):
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()[:120]
    row = find_one("scans", lambda s: s.get("id") == scan_id)
    if not row:
        return jsonify({"detail": "Scan not found"}), 404
    if user["role"] != "admin" and row.get("uploaded_by") != user["id"]:
        return jsonify({"detail": "Access denied"}), 403
    update("scans", scan_id, {"label": label or None})
    return jsonify({"ok": True})


@api.delete("/scans/<int:scan_id>")
@require_auth
def delete_scan(user, scan_id):
    row = find_one("scans", lambda s: s.get("id") == scan_id)
    if not row:
        return jsonify({"detail": "Scan not found"}), 404
    if user["role"] != "admin" and row.get("uploaded_by") != user["id"]:
        return jsonify({"detail": "Access denied"}), 403
    delete_by_id("scans", scan_id)
    return jsonify({"ok": True})


@api.get("/scans/<int:scan_id>/chats")
@require_auth
def get_scan_chats(user, scan_id):
    row = find_one("scans", lambda s: s.get("id") == scan_id)
    if not row:
        return jsonify({"detail": "Scan not found"}), 404
    if user["role"] != "admin" and row.get("uploaded_by") != user["id"]:
        return jsonify({"detail": "Access denied"}), 403
    rows = find_all(
        "scan_chats",
        where=lambda c: c.get("scan_id") == scan_id,
        order_by=lambda c: c.get("created") or "",
    )
    return jsonify([{"role": r["role"], "content": r["content"], "created": r.get("created")} for r in rows])


@api.delete("/scans/<int:scan_id>/chats")
@require_auth
def clear_scan_chats(user, scan_id):
    row = find_one("scans", lambda s: s.get("id") == scan_id)
    if not row:
        return jsonify({"detail": "Scan not found"}), 404
    if user["role"] != "admin" and row.get("uploaded_by") != user["id"]:
        return jsonify({"detail": "Access denied"}), 403
    delete_where("scan_chats", lambda c: c.get("scan_id") == scan_id)
    return jsonify({"ok": True})
