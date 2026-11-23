"""
Scanova - Breast Cancer AI
==========================
Main entry point.  Run:  python app.py
Then open:               http://localhost:5000
"""
import io
import sys

from flask import Flask, send_from_directory
from flask_cors import CORS

from backend.config import config, validate_security
from backend.middleware.security import register_security
from backend.storage.json_store import init_storage
from backend.routes import api


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(config.FRONTEND_DIR),
        static_url_path="",
    )
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["JSON_SORT_KEYS"] = False

    CORS(
        app,
        origins=config.CORS_ORIGINS,
        supports_credentials=True,
        allow_headers=["Authorization", "Content-Type"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    register_security(app)
    app.register_blueprint(api)
    init_storage()

    @app.get("/")
    def index():
        resp = send_from_directory(config.FRONTEND_DIR, "index.html")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    return app


app = create_app()


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backend.inference import models_status

    st = models_status()
    ok = "[OK]"
    mis = "[MISSING - copy to models/]"

    print()
    print("=" * 60)
    print("  Scanova - Breast Cancer AI")
    print("=" * 60)
    print(f"  CBIS model  : {ok if st['cbis'] else mis}")
    print(f"  RSNA model  : {ok if st['rsna'] else mis}")
    print(f"  BUSI model  : {ok if st['busi'] else mis}")
    print(f"  MRI model   : {ok if st['mri'] else mis}")
    print(f"  Fusion V3   : {ok if st['fusion_v3'] else mis}")
    print(f"  Fusion V2   : {ok if st['fusion_v2'] else mis}")
    print("=" * 60)
    from backend.services.llm_service import groq_configured, ollama_configured
    from backend.config import config as cfg
    groq_s = "[OK]" if groq_configured() else "[not set — will use Ollama if available]"
    ollama_s = (
        f"[OK] fallback {cfg.OLLAMA_MODEL} @ {cfg.OLLAMA_BASE_URL}"
        if ollama_configured()
        else "[disabled]"
    )
    print(f"  Groq LLM    : {groq_s}")
    print(f"  Ollama      : {ollama_s}")
    print("=" * 60)
    print(f"  Storage     : JSON ({config.DATA_DIR})")
    print(f"  Admin email : {config.ADMIN_EMAIL}")
    print(f"  URL         : http://localhost:{config.FLASK_PORT}")
    print("=" * 60)

    warnings = validate_security()
    if warnings:
        print("  Security notices:")
        for w in warnings:
            print(f"    ! {w}")
        print("=" * 60)
    print()

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
