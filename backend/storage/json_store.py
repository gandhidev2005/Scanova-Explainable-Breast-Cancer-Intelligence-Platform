"""
JSON file storage layer — active persistence backend for Scanova.
Each logical table is stored as  data/<table>.json
"""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from passlib.context import CryptContext

from backend.config import config

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

TABLES = (
    "users",
    "scans",
    "patients",
    "appointments",
    "role_requests",
    "admin_logs",
    "scan_chats",
    "direct_chat_sessions",
    "direct_chat_messages",
)

_lock = threading.RLock()


def now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _table_path(table: str) -> Path:
    if table not in TABLES:
        raise ValueError(f"Unknown table: {table}")
    return config.DATA_DIR / f"{table}.json"


def _ensure_data_dir() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load(table: str) -> list[dict[str, Any]]:
    _ensure_data_dir()
    path = _table_path(table)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def _save(table: str, records: list[dict[str, Any]]) -> None:
    _ensure_data_dir()
    path = _table_path(table)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def next_id(table: str) -> int:
    with _lock:
        records = _load(table)
        if not records:
            return 1
        return max(int(r.get("id", 0)) for r in records) + 1


def find_all(
    table: str,
    *,
    where: Callable[[dict[str, Any]], bool] | None = None,
    order_by: Callable[[dict[str, Any]], Any] | None = None,
    reverse: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with _lock:
        rows = [deepcopy(r) for r in _load(table)]
    if where:
        rows = [r for r in rows if where(r)]
    if order_by:
        rows.sort(key=order_by, reverse=reverse)
    if limit is not None:
        rows = rows[:limit]
    return rows


def find_one(
    table: str,
    where: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    rows = find_all(table, where=where, limit=1)
    return rows[0] if rows else None


def insert(table: str, data: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        records = _load(table)
        row = deepcopy(data)
        row.setdefault("id", next_id(table))
        row.setdefault("created", now_str())
        records.append(row)
        _save(table, records)
        return deepcopy(row)


def update(table: str, record_id: int, updates: dict[str, Any]) -> dict[str, Any] | None:
    with _lock:
        records = _load(table)
        for i, row in enumerate(records):
            if int(row.get("id", 0)) == int(record_id):
                merged = {**row, **updates, "id": row["id"]}
                records[i] = merged
                _save(table, records)
                return deepcopy(merged)
    return None


def delete_by_id(table: str, record_id: int) -> bool:
    with _lock:
        records = _load(table)
        new_records = [r for r in records if int(r.get("id", 0)) != int(record_id)]
        if len(new_records) == len(records):
            return False
        _save(table, new_records)
        return True


def delete_where(table: str, where: Callable[[dict[str, Any]], bool]) -> int:
    with _lock:
        records = _load(table)
        kept = [r for r in records if not where(r)]
        removed = len(records) - len(kept)
        if removed:
            _save(table, kept)
        return removed


def count(table: str, where: Callable[[dict[str, Any]], bool] | None = None) -> int:
    return len(find_all(table, where=where))


def init_storage() -> None:
    """Create empty JSON tables and seed the admin account."""
    _ensure_data_dir()
    with _lock:
        for table in TABLES:
            path = _table_path(table)
            if not path.exists():
                _save(table, [])

        admin = find_one("users", lambda u: u.get("email") == config.ADMIN_EMAIL)
        if not admin:
            insert(
                "users",
                {
                    "name": config.ADMIN_NAME,
                    "email": config.ADMIN_EMAIL,
                    "password": pwd_ctx.hash(config.ADMIN_PASSWORD),
                    "role": "admin",
                    "mfa_enabled": 0,
                    "mfa_secret": None,
                },
            )


# Backwards-compatible aliases used by legacy imports
init_db = init_storage
get_db = None  # SQLite removed — use json_store helpers instead
