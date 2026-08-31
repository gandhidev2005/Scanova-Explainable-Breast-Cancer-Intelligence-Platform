"""Patient registry routes."""

from flask import jsonify, request

from backend.middleware.auth import require_auth
from backend.routes import api
from backend.storage.json_store import delete_by_id, find_all, insert


@api.get("/patients")
@require_auth
def get_patients(user):
    if user["role"] not in ["clinic", "radiologist", "admin"]:
        return jsonify({"detail": "Access denied"}), 403
    rows = find_all(
        "patients",
        order_by=lambda p: p.get("created") or "",
        reverse=True,
    )
    return jsonify(rows)


@api.post("/patients")
@require_auth
def add_patient(user):
    if user["role"] not in ["clinic", "admin"]:
        return jsonify({"detail": "Access denied"}), 403
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"detail": "Patient name is required"}), 400
    insert(
        "patients",
        {
            "name": name,
            "dob": data.get("dob"),
            "email": data.get("email"),
            "phone": data.get("phone"),
            "risk_level": data.get("risk_level", "low"),
            "notes": data.get("notes"),
        },
    )
    return jsonify({"message": "Patient added"}), 201


@api.delete("/patients/<int:pid>")
@require_auth
def delete_patient(user, pid):
    if user["role"] not in ["clinic", "admin"]:
        return jsonify({"detail": "Access denied"}), 403
    delete_by_id("patients", pid)
    return jsonify({"message": "Patient removed"})
