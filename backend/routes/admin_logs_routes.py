"""Admin audit log routes."""

from flask import jsonify

from backend.middleware.auth import require_role
from backend.routes import api
from backend.storage.json_store import find_all


@api.get("/admin-logs")
@require_role("admin")
def get_admin_logs(user):
    rows = find_all(
        "admin_logs",
        order_by=lambda l: l.get("created") or "",
        reverse=True,
        limit=200,
    )
    return jsonify(rows)
