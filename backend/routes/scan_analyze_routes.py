"""Scan upload and AI analysis."""

import os

from flask import jsonify, request

from backend.config import config
from backend.inference import run_inference
from backend.middleware.auth import require_auth
from backend.routes import api
from backend.services.image_service import load_image_from_upload
from backend.storage.json_store import insert

_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}


@api.post("/scan/analyze")
@require_auth
def analyze_scan(user):
    files = []
    try:
        files = request.files.getlist("file") or []
    except Exception:
        files = []
    if not files and "file" in request.files:
        files = [request.files["file"]]
    if not files:
        return jsonify({"detail": "No file provided"}), 400

    out = []
    for f in files:
        fname = os.path.basename((f.filename or "").strip())
        if not fname:
            return jsonify({"detail": "Invalid filename"}), 400

        ext = os.path.splitext(fname.lower())[1]
        if ext not in _ALLOWED_EXT:
            return jsonify(
                {
                    "detail": "Unsupported file type. Upload JPG, PNG, or DICOM (.dcm)."
                }
            ), 400

        contents = f.read()
        if len(contents) > config.MAX_CONTENT_LENGTH:
            return jsonify(
                {"detail": f"File too large (max {config.MAX_UPLOAD_MB} MB)"}
            ), 413
        if len(contents) == 0:
            return jsonify({"detail": "Empty file"}), 400

        try:
            img = load_image_from_upload(
                fname, contents, getattr(f, "mimetype", "") or ""
            )
        except ValueError as e:
            return jsonify({"detail": str(e)}), 400
        except Exception:
            return jsonify({"detail": "Cannot read image file"}), 400

        result = run_inference(img)
        row = insert(
            "scans",
            {
                "uploaded_by": user["id"],
                "filename": fname,
                "modality": "mammography",
                "prediction": result["prediction"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "uncertainty": result["uncertainty"],
                "risk_level": result["risk_level"],
                "heatmap": result["heatmap"],
                "shap_mammo": result["shap_cbis"],
                "shap_rsna": result["shap_rsna"],
                "shap_us": result["shap_us"],
            },
        )
        result["scan_id"] = row["id"]
        result["filename"] = fname
        out.append(result)

    return jsonify(out[0] if len(out) == 1 else out)
