"""Auth routes: login, register, profile, password."""

from flask import jsonify, request

from backend.middleware.auth import make_token, require_auth
from backend.middleware.rate_limit import rate_limit
from backend.middleware.security import sanitize_user
from backend.routes import api
from backend.services.password_service import validate_password
from backend.storage.json_store import find_one, insert, pwd_ctx, update


@api.post("/auth/login")
@rate_limit("auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    row = find_one("users", lambda u: u.get("email") == email)
    if not row or not pwd_ctx.verify(data.get("password", ""), row["password"]):
        return jsonify({"detail": "Invalid email or password"}), 401

    import pyotp

    mfa_enabled = int(row.get("mfa_enabled") or 0) == 1
    mfa_secret = (row.get("mfa_secret") or "").strip()
    if mfa_enabled:
        code = (data.get("mfa_code") or "").strip().replace(" ", "")
        if not code:
            return jsonify({"detail": "MFA code required", "mfa_required": True}), 401
        try:
            totp = pyotp.TOTP(mfa_secret)
            if not totp.verify(code, valid_window=1):
                return jsonify({"detail": "Invalid MFA code", "mfa_required": True}), 401
        except Exception:
            return jsonify({"detail": "Invalid MFA configuration"}), 400

    safe = sanitize_user(row)
    return jsonify(
        {
            "token": make_token(row["email"], row["role"]),
            "user": safe,
        }
    )


@api.post("/auth/register")
@rate_limit("auth/register", limit=5)
def register():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    pw = data.get("password", "")
    role = "patient"
    if not name or not email or not pw:
        return jsonify({"detail": "Name, email and password are required"}), 400

    pw_err = validate_password(pw)
    if pw_err:
        return jsonify({"detail": pw_err}), 400

    if find_one("users", lambda u: u.get("email") == email):
        return jsonify({"detail": "Email already registered"}), 400

    row = insert(
        "users",
        {
            "name": name,
            "email": email,
            "password": pwd_ctx.hash(pw),
            "role": role,
            "mfa_enabled": 0,
            "mfa_secret": None,
        },
    )
    return jsonify(
        {
            "token": make_token(row["email"], row["role"]),
            "user": sanitize_user(row),
        }
    )


@api.get("/auth/me")
@require_auth
def me(user):
    return jsonify(sanitize_user(user))


@api.put("/auth/profile")
@require_auth
def update_profile(user):
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    if not name or not email:
        return jsonify({"detail": "Name and email are required"}), 400
    other = find_one(
        "users", lambda u: u.get("email") == email and u.get("id") != user["id"]
    )
    if other:
        return jsonify({"detail": "Email already in use"}), 400
    update("users", user["id"], {"name": name, "email": email})
    return jsonify({"message": "Profile updated", "name": name, "email": email})


@api.put("/auth/password")
@require_auth
def change_password(user):
    data = request.get_json(silent=True) or {}
    current_pw = data.get("current_password", "")
    new_pw = data.get("new_password", "")
    if not pwd_ctx.verify(current_pw, user["password"]):
        return jsonify({"detail": "Current password is incorrect"}), 400
    pw_err = validate_password(new_pw)
    if pw_err:
        return jsonify({"detail": pw_err}), 400
    update("users", user["id"], {"password": pwd_ctx.hash(new_pw)})
    return jsonify({"message": "Password updated successfully"})
