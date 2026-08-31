"""MFA (TOTP) routes."""

import base64
import io

import pyotp
import qrcode
from flask import jsonify, request

from backend.middleware.auth import require_auth
from backend.routes import api
from backend.storage.json_store import find_one, pwd_ctx, update


@api.get("/auth/mfa/status")
@require_auth
def mfa_status(user):
    row = find_one("users", lambda u: u.get("id") == user["id"])
    enabled = int(row.get("mfa_enabled") or 0) == 1 if row else False
    return jsonify({"enabled": enabled})


@api.post("/auth/mfa/start")
@require_auth
def mfa_start(user):
    secret = pyotp.random_base32()
    issuer = "Scanova"
    label = f"{issuer}:{user.get('email')}"
    uri = pyotp.TOTP(secret).provisioning_uri(name=label, issuer_name=issuer)

    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    update("users", user["id"], {"mfa_secret": secret, "mfa_enabled": 0})
    return jsonify({"secret": secret, "otpauth_uri": uri, "qr_png_base64": qr_b64})


@api.post("/auth/mfa/verify")
@require_auth
def mfa_verify(user):
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip().replace(" ", "")
    if not code:
        return jsonify({"detail": "Code required"}), 400

    row = find_one("users", lambda u: u.get("id") == user["id"])
    if not row or not (row.get("mfa_secret") or "").strip():
        return jsonify({"detail": "Start MFA setup first"}), 400

    secret = (row["mfa_secret"] or "").strip()
    try:
        ok = bool(pyotp.TOTP(secret).verify(code, valid_window=1))
    except Exception:
        ok = False
    if not ok:
        return jsonify({"detail": "Invalid code"}), 400

    update("users", user["id"], {"mfa_enabled": 1})
    return jsonify({"enabled": True})


@api.post("/auth/mfa/disable")
@require_auth
def mfa_disable(user):
    data = request.get_json(silent=True) or {}
    pw = data.get("password", "")
    code = (data.get("code") or "").strip().replace(" ", "")
    if not pw or not code:
        return jsonify({"detail": "Password and code required"}), 400

    row = find_one("users", lambda u: u.get("id") == user["id"])
    if not row:
        return jsonify({"detail": "Not found"}), 404
    if not pwd_ctx.verify(pw, row["password"]):
        return jsonify({"detail": "Current password is incorrect"}), 400
    if int(row.get("mfa_enabled") or 0) != 1:
        return jsonify({"detail": "MFA is not enabled"}), 400

    secret = (row.get("mfa_secret") or "").strip()
    try:
        if not pyotp.TOTP(secret).verify(code, valid_window=1):
            return jsonify({"detail": "Invalid code"}), 400
    except Exception:
        return jsonify({"detail": "Invalid MFA configuration"}), 400

    update("users", user["id"], {"mfa_enabled": 0, "mfa_secret": None})
    return jsonify({"enabled": False})
