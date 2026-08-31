"""Compatibility shim — use backend.storage.json_store instead."""

from backend.storage.json_store import init_db, pwd_ctx

__all__ = ["init_db", "pwd_ctx"]
