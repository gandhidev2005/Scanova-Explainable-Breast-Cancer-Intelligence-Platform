"""Appointment / schedule routes."""

from flask import jsonify, request

from backend.middleware.auth import require_auth
from backend.routes import api
from backend.storage.json_store import delete_by_id, find_all, insert, update


@api.get("/schedule")
@require_auth
def get_schedule(user):
    rows = find_all(
        "appointments",
        order_by=lambda a: (a.get("date") or "", a.get("time") or ""),
        reverse=True,
    )
    return jsonify(rows)


@api.post("/schedule")
@require_auth
def add_appointment(user):
    data = request.get_json(silent=True) or {}
    patient_name = data.get("patient_name", "").strip()
    date = data.get("date", "").strip()
    time = data.get("time", "").strip()
    if not patient_name or not date or not time:
        return jsonify({"detail": "Patient name, date and time are required"}), 400
    insert(
        "appointments",
        {
            "patient_name": patient_name,
            "date": date,
            "time": time,
            "type": data.get("type", "Mammography"),
            "doctor": data.get("doctor", ""),
            "status": data.get("status", "scheduled"),
            "notes": data.get("notes", ""),
            "created_by": user["id"],
        },
    )
    return jsonify({"message": "Appointment scheduled"}), 201


@api.delete("/schedule/<int:appt_id>")
@require_auth
def delete_appointment(user, appt_id):
    delete_by_id("appointments", appt_id)
    return jsonify({"message": "Appointment removed"})


@api.put("/schedule/<int:appt_id>")
@require_auth
def update_appointment_status(user, appt_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status", "scheduled")
    update("appointments", appt_id, {"status": status})
    return jsonify({"message": "Status updated"})
