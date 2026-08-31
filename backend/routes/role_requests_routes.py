"""Role upgrade request workflow."""

from flask import jsonify, request

from backend.middleware.auth import require_auth, require_role
from backend.routes import api
from backend.services.admin_log_service import log_admin_action
from backend.storage.json_store import count, find_all, find_one, insert, now_str, update


@api.post("/role-requests")
@require_auth
def submit_role_request(user):
    data = request.get_json(silent=True) or {}
    req_r = data.get("requested_role", "").strip().lower()
    valid = {"radiologist", "clinic", "admin"}
    if req_r not in valid:
        return jsonify({"detail": "Invalid role requested"}), 400
    if req_r == user["role"]:
        return jsonify({"detail": "You already have this role"}), 400

    existing = find_one(
        "role_requests",
        lambda r: r.get("user_id") == user["id"] and r.get("status") == "pending",
    )
    if existing:
        return jsonify({"detail": "You already have a pending role request"}), 400

    insert(
        "role_requests",
        {
            "user_id": user["id"],
            "current_role": user["role"],
            "requested_role": req_r,
            "reason": data.get("reason", "").strip(),
            "status": "pending",
        },
    )
    return jsonify({"message": "Role upgrade request submitted"}), 201


@api.get("/role-requests/my")
@require_auth
def my_role_requests(user):
    rows = find_all(
        "role_requests",
        where=lambda r: r.get("user_id") == user["id"],
        order_by=lambda r: r.get("created") or "",
        reverse=True,
        limit=5,
    )
    return jsonify(rows)


@api.get("/role-requests")
@require_role("admin")
def get_role_requests(user):
    requests_rows = find_all(
        "role_requests",
        order_by=lambda r: r.get("created") or "",
        reverse=True,
    )
    users = {u["id"]: u for u in find_all("users")}
    out = []
    for rr in requests_rows:
        d = dict(rr)
        u = users.get(rr.get("user_id"))
        d["user_name"] = u["name"] if u else None
        d["user_email"] = u["email"] if u else None
        out.append(d)
    return jsonify(out)


@api.put("/role-requests/<int:req_id>")
@require_role("admin")
def resolve_role_request(user, req_id):
    data = request.get_json(silent=True) or {}
    action = data.get("action", "").lower()
    notes = data.get("notes", "").strip()
    if action not in ("approve", "reject"):
        return jsonify({"detail": "Action must be 'approve' or 'reject'"}), 400

    rr = find_one("role_requests", lambda r: r.get("id") == req_id)
    if not rr:
        return jsonify({"detail": "Request not found"}), 404
    if rr.get("status") != "pending":
        return jsonify({"detail": "Request is already resolved"}), 400

    new_status = "approved" if action == "approve" else "rejected"
    update(
        "role_requests",
        req_id,
        {
            "status": new_status,
            "admin_notes": notes,
            "resolved_at": now_str(),
            "resolved_by": user["id"],
        },
    )

    target = find_one("users", lambda u: u.get("id") == rr["user_id"])
    if action == "approve":
        update("users", rr["user_id"], {"role": rr["requested_role"]})

    tname = target["name"] if target else f"User #{rr['user_id']}"
    log_admin_action(
        user,
        f"Role Request {'Approved' if action == 'approve' else 'Rejected'}",
        rr["user_id"],
        tname,
        f"{rr['current_role']} → {rr['requested_role']}"
        + (f" | Notes: {notes}" if notes else ""),
    )
    return jsonify({"message": f"Request {new_status}"})


@api.get("/role-requests/pending-count")
@require_role("admin")
def pending_count(user):
    n = count("role_requests", where=lambda r: r.get("status") == "pending")
    return jsonify({"count": n})
