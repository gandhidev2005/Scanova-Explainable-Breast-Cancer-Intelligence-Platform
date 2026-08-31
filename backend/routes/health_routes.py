"""Health check endpoint."""

from flask import jsonify

from backend.inference import models_status
from backend.routes import api


@api.get("/health")
def health():
    st = models_status()
    return jsonify({"status": "ok", **st})
