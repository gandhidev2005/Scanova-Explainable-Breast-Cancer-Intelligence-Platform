"""Password strength validation."""

import re

from backend.config import config


def validate_password(password: str) -> str | None:
    """Return error message if invalid, else None."""
    if not password:
        return "Password is required"
    if len(password) < config.MIN_PASSWORD_LENGTH:
        return f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters"
    if not re.search(r"[A-Za-z]", password):
        return "Password must contain at least one letter"
    if not re.search(r"\d", password):
        return "Password must contain at least one number"
    return None
