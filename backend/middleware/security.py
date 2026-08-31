"""HTTP security headers and request hardening."""

from flask import Flask, request

from backend.config import config

SENSITIVE_USER_KEYS = frozenset({"password", "mfa_secret"})


def sanitize_user(user: dict | None) -> dict | None:
    if not user:
        return user
    return {k: v for k, v in user.items() if k not in SENSITIVE_USER_KEYS}


def register_security(app: Flask) -> None:
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        response.headers["Cache-Control"] = response.headers.get(
            "Cache-Control", "no-store"
        )
        if not config.FLASK_DEBUG:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response

    @app.before_request
    def block_trace():
        if request.method == "TRACE":
            return ("Method Not Allowed", 405)
