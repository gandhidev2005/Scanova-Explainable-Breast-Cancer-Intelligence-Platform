"""
Application configuration — all secrets and model IDs loaded from .env only.
No API keys, passwords, or LLM model names are hardcoded in source.
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
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env(name: str, default: str = "") -> str:
    """Read a string from .env only — never embed secrets/models in code."""
    return (os.getenv(name) or default).strip()


def _require_env(name: str, *, dev_fallback: str | None = None) -> str:
    value = _env(name)
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
    FLASK_HOST = _env("FLASK_HOST", "127.0.0.1")
    FLASK_PORT = int(_env("FLASK_PORT", "5000") or "5000")
    FLASK_DEBUG = _bool("FLASK_DEBUG", False)
    ALLOW_INSECURE_DEFAULTS = _bool("ALLOW_INSECURE_DEFAULTS", False)

    SECRET_KEY = _require_env(
        "SCANOVA_SECRET",
        # Random per-process only when ALLOW_INSECURE_DEFAULTS=true and unset —
        # never a fixed secret string in source.
        dev_fallback=secrets.token_urlsafe(32) if _bool("ALLOW_INSECURE_DEFAULTS") else None,
    )
    JWT_ALGORITHM = _env("JWT_ALGORITHM", "HS256")
    JWT_TOKEN_HOURS = int(_env("JWT_TOKEN_HOURS", "24") or "24")

    ADMIN_EMAIL = _env("ADMIN_EMAIL").lower()
    # Password must come from .env — no hardcoded password in source
    ADMIN_PASSWORD = _require_env("ADMIN_PASSWORD")
    ADMIN_NAME = _env("ADMIN_NAME", "Administrator")

    # LLM — values only from .env (see .env.example for names to set)
    GROQ_API_KEY = _env("GROQ_API_KEY")
    GROQ_API_URL = _env("GROQ_API_URL")
    GROQ_MODEL = _env("GROQ_MODEL")

    OLLAMA_ENABLED = _bool("OLLAMA_ENABLED", False)
    OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL").rstrip("/")
    OLLAMA_MODEL = _env("OLLAMA_MODEL")

    STORAGE_BACKEND = _env("STORAGE_BACKEND", "json").lower()
    DATA_DIR = ROOT / (_env("DATA_DIR", "data") or "data")

    MODELS_DIR = ROOT / (_env("MODELS_DIR", "models") or "models")
    FRONTEND_DIR = ROOT / (_env("FRONTEND_DIR", "frontend") or "frontend")

    MYSQL_HOST = _env("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = int(_env("MYSQL_PORT", "3306") or "3306")
    MYSQL_USER = _env("MYSQL_USER", "scanova")
    MYSQL_PASSWORD = _env("MYSQL_PASSWORD")
    MYSQL_DATABASE = _env("MYSQL_DATABASE", "scanova")

    CORS_ORIGINS = [
        o.strip()
        for o in _env(
            "CORS_ORIGINS",
            "http://127.0.0.1:5000,http://localhost:5000",
        ).split(",")
        if o.strip()
    ]
    MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "16") or "16")
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    RATE_LIMIT_LOGIN = int(_env("RATE_LIMIT_LOGIN", "10") or "10")
    RATE_LIMIT_WINDOW_SEC = int(_env("RATE_LIMIT_WINDOW_SEC", "900") or "900")
    MIN_PASSWORD_LENGTH = int(_env("MIN_PASSWORD_LENGTH", "8") or "8")


config = Config()


def validate_security() -> list[str]:
    """Return security warnings (printed at startup)."""
    issues: list[str] = []

    if config.SECRET_KEY.lower() in _WEAK_SECRETS or len(config.SECRET_KEY) < 32:
        issues.append(
            "SCANOVA_SECRET is weak or too short — use at least 32 random characters."
        )

    if not config.ADMIN_EMAIL:
        issues.append("ADMIN_EMAIL is empty in .env.")

    if config.ADMIN_PASSWORD.lower() in _WEAK_ADMIN_PASSWORDS:
        issues.append("ADMIN_PASSWORD is a commonly used weak password.")

    if not config.GROQ_API_KEY or config.GROQ_API_KEY.startswith("your-"):
        if config.OLLAMA_ENABLED and config.OLLAMA_MODEL and config.OLLAMA_BASE_URL:
            issues.append(
                "GROQ_API_KEY not set in .env — AI chat will use Ollama fallback "
                f"({config.OLLAMA_MODEL})."
            )
        else:
            issues.append(
                "GROQ_API_KEY not set and Ollama is not fully configured in .env — "
                "AI chat will be unavailable."
            )
    elif not config.GROQ_MODEL or not config.GROQ_API_URL:
        issues.append("GROQ_MODEL / GROQ_API_URL must be set in .env when using Groq.")

    if config.OLLAMA_ENABLED and (not config.OLLAMA_MODEL or not config.OLLAMA_BASE_URL):
        issues.append(
            "OLLAMA_ENABLED=true but OLLAMA_MODEL / OLLAMA_BASE_URL missing in .env."
        )

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
