"""Admin audit logging."""

from backend.storage.json_store import insert


def log_admin_action(admin_user, action, target_id=None, target_name=None, details=None):
    insert(
        "admin_logs",
        {
            "admin_id": admin_user["id"],
            "admin_name": admin_user["name"],
            "action": action,
            "target_id": target_id,
            "target_name": target_name,
            "details": details,
        },
    )
