"""In-memory rate limiting by client IP + route key."""

import time
from collections import defaultdict
from functools import wraps

from flask import jsonify, request

from backend.config import config

_buckets: dict[str, list[float]] = defaultdict(list)


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_key(route: str) -> str:
    return f"{_client_ip()}:{route}"


def _is_rate_limited(key: str, limit: int, window_sec: int) -> bool:
    now = time.time()
    cutoff = now - window_sec
    hits = [t for t in _buckets[key] if t > cutoff]
    if len(hits) >= limit:
        _buckets[key] = hits
        return True
    hits.append(now)
    _buckets[key] = hits
    return False


def rate_limit(route: str, *, limit: int | None = None, window_sec: int | None = None):
    """Decorator — returns 429 when limit exceeded."""
    lim = limit if limit is not None else config.RATE_LIMIT_LOGIN
    win = window_sec if window_sec is not None else config.RATE_LIMIT_WINDOW_SEC

    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            key = _rate_key(route)
            if _is_rate_limited(key, lim, win):
                return jsonify(
                    {"detail": "Too many attempts. Please try again later."}
                ), 429
            return f(*args, **kwargs)

        return wrapped

    return decorator
