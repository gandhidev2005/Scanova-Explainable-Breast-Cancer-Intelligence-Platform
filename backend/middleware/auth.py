"""JWT authentication middleware."""

from datetime import datetime, timedelta
from functools import wraps

from flask import jsonify, request
from jose import JWTError, jwt

from backend.config import config
from backend.storage.json_store import find_one


def make_token(email: str, role: str) -> str:
    exp = datetime.utcnow() + timedelta(hours=config.JWT_TOKEN_HOURS)
    return jwt.encode(
        {"sub": email, "role": role, "exp": exp},
        config.SECRET_KEY,
        algorithm=config.JWT_ALGORITHM,
    )


def _get_user_from_token():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(
            token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM]
        )
        email = payload.get("sub")
        if not email:
            return None
        row = find_one("users", lambda u: u.get("email") == email)
        return row
    except JWTError:
        return None


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = _get_user_from_token()
        if not user:
            return jsonify({"detail": "Authentication required"}), 401
        return f(user, *args, **kwargs)

    return decorated


def require_role(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = _get_user_from_token()
            if not user:
                return jsonify({"detail": "Authentication required"}), 401
            if user["role"] not in roles:
                return jsonify({"detail": "Access denied"}), 403
            return f(user, *args, **kwargs)

        return decorated

    return decorator
