"""Admin user management routes."""

from flask import jsonify, request

from backend.middleware.auth import require_role
from backend.routes import api
from backend.services.admin_log_service import log_admin_action
from backend.storage.json_store import (
    count,
    delete_by_id,
    find_all,
    find_one,
    now_str,
    update,
)


@api.get("/users")
@require_role("admin")
def get_users(user):
    rows = find_all(
        "users",
        order_by=lambda u: u.get("created") or "",
        reverse=True,
    )
    return jsonify(
        [
            {
                "id": r["id"],
                "name": r["name"],
                "email": r["email"],
                "role": r["role"],
                "created": r.get("created"),
            }
            for r in rows
        ]
    )


@api.delete("/users/<int:user_id>")
@require_role("admin")
def delete_user(user, user_id):
    if user_id == user["id"]:
        return jsonify({"detail": "Cannot delete your own account"}), 400
    target = find_one("users", lambda u: u.get("id") == user_id)
    delete_by_id("users", user_id)
    if target:
        log_admin_action(
            user,
            "User Deleted",
            user_id,
            target["name"],
            f"Removed account {target['email']}",
        )
    return jsonify({"message": "User deleted"})


@api.put("/users/<int:user_id>/role")
@require_role("admin")
def change_user_role(user, user_id):
    if user_id == user["id"]:
        return jsonify({"detail": "Use Settings to change your own role"}), 400
    data = request.get_json(silent=True) or {}
    new_role = data.get("role", "").strip().lower()
    valid = {"patient", "radiologist", "clinic", "admin"}
    if new_role not in valid:
        return jsonify({"detail": f"Role must be one of: {', '.join(valid)}"}), 400

    target = find_one("users", lambda u: u.get("id") == user_id)
    if not target:
        return jsonify({"detail": "User not found"}), 404

    old_role = target["role"]
    update("users", user_id, {"role": new_role})

    pending = find_all(
        "role_requests",
        where=lambda r: r.get("user_id") == user_id and r.get("status") == "pending",
    )
    for rr in pending:
        update(
            "role_requests",
            rr["id"],
            {
                "status": "approved",
                "resolved_at": now_str(),
                "resolved_by": user["id"],
                "admin_notes": f"Role set directly to {new_role} by admin",
            },
        )

    log_admin_action(
        user,
        "Role Changed",
        user_id,
        target["name"],
        f"{old_role} → {new_role}",
    )
    return jsonify({"message": f"Role updated to {new_role}"})


@api.route("/users/<int:uid>/detail", methods=["GET"])
@require_role("admin")
def get_user_detail(user, uid):
    u = find_one("users", lambda x: x.get("id") == uid)
    if not u:
        return jsonify({"detail": "User not found"}), 404

    scans = find_all(
        "scans",
        where=lambda s: s.get("uploaded_by") == uid,
        order_by=lambda s: s.get("created") or "",
        reverse=True,
    )
    scans_l = [
        {
            "id": s["id"],
            "filename": s.get("filename"),
            "label": s.get("label"),
            "prediction": s.get("prediction"),
            "probability": s.get("probability"),
            "confidence": s.get("confidence"),
            "uncertainty": s.get("uncertainty"),
            "risk_level": s.get("risk_level"),
            "shap_mammo": s.get("shap_mammo"),
            "shap_rsna": s.get("shap_rsna"),
            "shap_us": s.get("shap_us"),
            "created": s.get("created"),
            "modality": s.get("modality"),
        }
        for s in scans
    ]

    reqs = find_all(
        "role_requests",
        where=lambda r: r.get("user_id") == uid,
        order_by=lambda r: r.get("created") or "",
        reverse=True,
    )
    reqs_l = [
        {
            "id": r["id"],
            "requested_role": r.get("requested_role"),
            "reason": r.get("reason"),
            "status": r.get("status"),
            "created": r.get("created"),
            "resolved_at": r.get("resolved_at"),
            "admin_notes": r.get("admin_notes"),
        }
        for r in reqs
    ]

    chat_count = count("scan_chats", where=lambda c: c.get("user_id") == uid)
    sessions = find_all(
        "direct_chat_sessions", where=lambda s: s.get("user_id") == uid
    )
    session_ids = {s["id"] for s in sessions}
    dc_count = count(
        "direct_chat_messages",
        where=lambda m: m.get("role") == "user" and m.get("session_id") in session_ids,
    )

    total = len(scans_l)
    return jsonify(
        {
            "user": {
                "id": u["id"],
                "name": u["name"],
                "email": u["email"],
                "role": u["role"],
                "created": u.get("created"),
            },
            "scans": scans_l,
            "role_requests": reqs_l,
            "stats": {
                "total_scans": total,
                "high_risk": sum(1 for s in scans_l if s["risk_level"] == "high"),
                "medium_risk": sum(1 for s in scans_l if s["risk_level"] == "medium"),
                "low_risk": sum(1 for s in scans_l if s["risk_level"] == "low"),
                "malignant": sum(1 for s in scans_l if s["prediction"] == "malignant"),
                "benign": sum(1 for s in scans_l if s["prediction"] == "benign"),
                "chat_msgs": chat_count,
                "ai_chats": dc_count,
            },
        }
    )
