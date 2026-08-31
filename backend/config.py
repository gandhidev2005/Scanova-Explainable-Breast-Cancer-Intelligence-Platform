"""
Application configuration — all secrets loaded from .env only.
No API keys or passwords are stored in source code.
"""
import os
import secrets
import warnings
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_WEAK_SECRETS = frozenset(
    {
        "",
        "scanova-jwt-secret-change-in-prod",
        "change-me",
        "secret",
        "dev",
    }
)
_WEAK_ADMIN_PASSWORDS = frozenset({"admin", "admin123", "password", "123456"})


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _require_env(name: str, *, dev_fallback: str | None = None) -> str:
    value = (os.getenv(name) or "").strip()
    if value:
        return value
    if _bool("ALLOW_INSECURE_DEFAULTS") and dev_fallback is not None:
        warnings.warn(
            f"{name} not set — using insecure dev fallback. "
            "Set ALLOW_INSECURE_DEFAULTS=false and configure .env for production.",
            stacklevel=2,
        )
        return dev_fallback
    raise RuntimeError(
        f"{name} is required in .env. Copy .env.example → .env and set it."
    )


class Config:
    ROOT = ROOT
    FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = _bool("FLASK_DEBUG", False)
    ALLOW_INSECURE_DEFAULTS = _bool("ALLOW_INSECURE_DEFAULTS", False)

    SECRET_KEY = _require_env(
        "SCANOVA_SECRET",
        dev_fallback=secrets.token_urlsafe(32) if _bool("ALLOW_INSECURE_DEFAULTS") else None,
    )
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_TOKEN_HOURS = int(os.getenv("JWT_TOKEN_HOURS", "24"))

    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@scanova.com").strip().lower()
    ADMIN_PASSWORD = _require_env(
        "ADMIN_PASSWORD",
        dev_fallback="Admin@123" if _bool("ALLOW_INSECURE_DEFAULTS") else None,
    )
    ADMIN_NAME = os.getenv("ADMIN_NAME", "Administrator")

    GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
    GROQ_API_URL = os.getenv(
        "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
    )
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "json").strip().lower()
    DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")

    MODELS_DIR = ROOT / os.getenv("MODELS_DIR", "models")
    FRONTEND_DIR = ROOT / os.getenv("FRONTEND_DIR", "frontend")

    MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "scanova")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "scanova")

    # Security
    CORS_ORIGINS = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://127.0.0.1:5000,http://localhost:5000",
        ).split(",")
        if o.strip()
    ]
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "16"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    RATE_LIMIT_LOGIN = int(os.getenv("RATE_LIMIT_LOGIN", "10"))
    RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "900"))
    MIN_PASSWORD_LENGTH = int(os.getenv("MIN_PASSWORD_LENGTH", "8"))


config = Config()


def validate_security() -> list[str]:
    """Return security warnings (printed at startup)."""
    issues: list[str] = []

    if config.SECRET_KEY.lower() in _WEAK_SECRETS or len(config.SECRET_KEY) < 32:
        issues.append(
            "SCANOVA_SECRET is weak or too short — use at least 32 random characters."
        )

    if config.ADMIN_PASSWORD.lower() in _WEAK_ADMIN_PASSWORDS:
        issues.append("ADMIN_PASSWORD is a commonly used weak password.")

    if not config.GROQ_API_KEY or config.GROQ_API_KEY.startswith("your-"):
        issues.append("GROQ_API_KEY not configured — AI chat endpoints will be disabled.")

    if config.FLASK_DEBUG:
        issues.append("FLASK_DEBUG=true — never enable in production.")

    if config.ALLOW_INSECURE_DEFAULTS:
        issues.append("ALLOW_INSECURE_DEFAULTS=true — disable for production deployments.")

    if config.FLASK_HOST == "0.0.0.0" and not config.FLASK_DEBUG:
        issues.append(
            "FLASK_HOST=0.0.0.0 exposes the server on all interfaces — "
            "use 127.0.0.1 behind a reverse proxy when possible."
        )

    return issues
