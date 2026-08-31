"""
API blueprint — registers all domain route modules.
"""

from flask import Blueprint

api = Blueprint("api", __name__, url_prefix="/api")

# Import route modules for side-effect registration on `api`
from backend.routes import (  # noqa: E402, F401
    admin_logs_routes,
    auth_routes,
    dashboard_routes,
    direct_chat_routes,
    general_chat_routes,
    health_routes,
    mfa_routes,
    patients_routes,
    role_requests_routes,
    scan_analyze_routes,
    scan_chat_routes,
    schedule_routes,
    scans_routes,
    users_routes,
)
