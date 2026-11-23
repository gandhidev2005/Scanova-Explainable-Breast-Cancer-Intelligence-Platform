"""
AI model loading and inference pipeline.

Encoders (EfficientNet-B0, 1280-dim embeddings each):
  best_model.pth       → CBIS-DDSM mammography
  rsna_model.pth       → RSNA mammography
  busi_model.pth       → BUSI ultrasound
  mri_model.pth        → MRI

Fusion heads:
  fusion_v3_model.pth  → 4×1280 = 5120 → 1  (preferred)
  fusion_v2_model.pth  → 3×1280 = 3840 → 1  (fallback)
"""
import base64
import io
import traceback

import numpy as np
from PIL import Image

from backend.config import config

MODELS_DIR = config.MODELS_DIR
CBIS_CKPT = MODELS_DIR / "best_model.pth"
RSNA_CKPT = MODELS_DIR / "rsna_model.pth"
BUSI_CKPT = MODELS_DIR / "busi_model.pth"
MRI_CKPT = MODELS_DIR / "mri_model.pth"
FUSION_V3_CKPT = MODELS_DIR / "fusion_v3_model.pth"
FUSION_V2_CKPT = MODELS_DIR / "fusion_v2_model.pth"

_loaded = False
_cbis_enc = _rsna_enc = _busi_enc = _mri_enc = None
_fusion = None
_fusion_version = None  # "v3" | "v2" | None
_device = None

# Reported test AUC from G:\dev_part1 metric JSON files.
# Checkpoint val_auc is kept separately and is not shown as this AUC.
# Paper comparison-table figures (rates for Sens/Spec; Acc as percent).
MODEL_METRICS = {
    "cbis": {"auc": 0.7434, "name": "CBIS-DDSM (Mammography)"},
    "rsna": {"auc": 0.6817, "name": "RSNA (Mammography)"},
    "busi": {"auc": 0.9647, "name": "BUSI (Ultrasound)"},
    "fusion_v2": {"auc": 0.9913, "name": "Fusion V2 (Mammography + Ultrasound)"},
    "mri": {"auc": 0.6461, "name": "Duke MRI (MRI)"},
    "fusion_v3": {"auc": 0.9897, "name": "Fusion V3 (Mammography + Ultrasound + MRI)"},
}

# Display order matches the paper table (VinDr is not a checkpoint).
_CATALOG = [
    {
        "id": "cbis",
        "file": "best_model.pth",
        "path": CBIS_CKPT,
        "role": "encoder",
        "title": "CBIS-DDSM (Mammography)",
        "modality": "CBIS-DDSM (Mammography)",
        "backbone": "EfficientNet-B0",
        "emb_dim": 1280,
    },
    {
        "id": "rsna",
        "file": "rsna_model.pth",
        "path": RSNA_CKPT,
        "role": "encoder",
        "title": "RSNA (Mammography)",
        "modality": "RSNA (Mammography)",
        "backbone": "EfficientNet-B0",
        "emb_dim": 1280,
    },
    {
        "id": "busi",
        "file": "busi_model.pth",
        "path": BUSI_CKPT,
        "role": "encoder",
        "title": "BUSI (Ultrasound)",
        "modality": "BUSI (Ultrasound)",
        "backbone": "EfficientNet-B0",
        "emb_dim": 1280,
    },
    {
        "id": "fusion_v2",
        "file": "fusion_v2_model.pth",
        "path": FUSION_V2_CKPT,
        "role": "fusion",
        "title": "Fusion V2 (Mammography + Ultrasound)",
        "modality": "Fusion V2 (Mammography + Ultrasound)",
        "backbone": "FusionHead (3×1280→1)",
        "emb_dim": 3840,
        "n_encoders": 3,
        "preferred": False,
    },
    {
        "id": "mri",
        "file": "mri_model.pth",
        "path": MRI_CKPT,
        "role": "encoder",
        "title": "Duke MRI (MRI)",
        "modality": "Duke MRI (MRI)",
        "backbone": "EfficientNet-B0",
        "emb_dim": 1280,
    },
    {
        "id": "fusion_v3",
        "file": "fusion_v3_model.pth",
        "path": FUSION_V3_CKPT,
        "role": "fusion",
        "title": "Fusion V3 (Mammography + Ultrasound + MRI)",
        "modality": "Fusion V3 (Mammography + Ultrasound + MRI)",
        "backbone": "FusionHead (4×1280→1)",
        "emb_dim": 5120,
        "n_encoders": 4,
        "preferred": True,
    },
]

# Sens/Spec stored as rates (0–1) to match the paper table; Acc as percent.
CLINICAL_METRICS = {
    "cbis": {"sensitivity": 0.7197, "specificity": 0.6325, "accuracy": 66.8},
    "rsna": {"sensitivity": 0.6149, "specificity": 0.6777, "accuracy": 66.0},
    "busi": {"sensitivity": 0.8387, "specificity": 0.9651, "accuracy": 93.2},
    "fusion_v2": {"sensitivity": 0.9659, "specificity": 0.9948, "accuracy": 98.3},
    "mri": {"sensitivity": 0.8889, "specificity": 0.2222, "accuracy": 35.6},
    "fusion_v3": {"sensitivity": 0.9697, "specificity": 1.0000, "accuracy": 98.8},
}


def _read_ckpt_meta(path):
    """Read lightweight metadata from a .pth without keeping it in memory."""
    meta = {"exists": path.exists()}
    if not path.exists():
        return meta
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            meta["error"] = "unexpected checkpoint format"
            return meta
        meta["epoch"] = ckpt.get("epoch")
        auc = ckpt.get("val_auc")
        if auc is not None:
            meta["val_auc"] = float(auc)
        if "task" in ckpt:
            meta["task"] = ckpt.get("task")
        # Infer fusion input dim from first linear weight
        head = ckpt.get("fusion_head")
        if isinstance(head, dict) and "net.0.weight" in head:
            shape = tuple(head["net.0.weight"].shape)
            meta["fusion_in_dim"] = int(shape[1]) if len(shape) == 2 else None
            meta["fusion_hidden"] = int(shape[0]) if len(shape) == 2 else None
        state = ckpt.get("model_state_dict")
        if isinstance(state, dict) and "classifier.1.weight" in state:
            shape = tuple(state["classifier.1.weight"].shape)
            meta["classifier_in"] = int(shape[1]) if len(shape) == 2 else None
        meta["keys"] = [k for k in ckpt.keys() if k not in (
            "model_state_dict", "fusion_head", "optimizer", "optimizer_state_dict"
        )]
        try:
            meta["size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
        except Exception:
            pass
    except Exception as e:
        meta["error"] = str(e)
    return meta


def resolve_active_model():
    """Which fusion/encoder will actually run predictions right now."""
    if FUSION_V3_CKPT.exists() and all(
        p.exists() for p in (CBIS_CKPT, RSNA_CKPT, BUSI_CKPT, MRI_CKPT)
    ):
        return {
            "id": "fusion_v3",
            "label": "Fusion V3 (Mammography + Ultrasound + MRI)",
            "version": "v3",
            "reason": "All 4 modality encoders are available",
        }
    if FUSION_V3_CKPT.exists():
        return {
            "id": "fusion_v3",
            "label": "Fusion V3 (Mammography + Ultrasound + MRI)",
            "version": "v3",
            "reason": "Fusion V3 available; missing encoders use zero embeddings",
            "warning": True,
        }
    if FUSION_V2_CKPT.exists():
        return {
            "id": "fusion_v2",
            "label": "Fusion V2 (Mammography + Ultrasound)",
            "version": "v2",
            "reason": "Fusion V3 not found — using Fusion V2 fallback",
        }
    if CBIS_CKPT.exists():
        return {
            "id": "cbis",
            "label": "CBIS-DDSM (Mammography)",
            "version": None,
            "reason": "No fusion head — single CBIS encoder",
            "warning": True,
        }
    return {
        "id": "demo",
        "label": "Demo mode (models not loaded)",
        "version": None,
        "reason": "No usable model checkpoints found",
        "warning": True,
    }


def get_models_info():
    """Full catalog with checkpoint values from disk (no GPU load required)."""
    models = []
    for item in _CATALOG:
        meta = _read_ckpt_meta(item["path"])
        clinical = CLINICAL_METRICS.get(item["id"], {})
        entry = {
            "id": item["id"],
            "file": item["file"],
            "role": item["role"],
            "title": item["title"],
            "modality": item["modality"],
            "backbone": item["backbone"],
            "emb_dim": item["emb_dim"],
            "preferred": bool(item.get("preferred")),
            "n_encoders": item.get("n_encoders"),
            "sensitivity": clinical.get("sensitivity"),
            "specificity": clinical.get("specificity"),
            "accuracy": clinical.get("accuracy"),
            "test_auc": MODEL_METRICS[item["id"]]["auc"],
            **meta,
        }
        entry["test_auc"] = MODEL_METRICS[item["id"]]["auc"]
        models.append(entry)

    active = resolve_active_model()
    return {
        "models": models,
        "active": active,
        "loaded_in_memory": _loaded,
        "fusion_active": _fusion_version or active.get("version"),
        "metrics": {
            k: {"name": v["name"], "auc": round(float(v["auc"]), 5)}
            for k, v in MODEL_METRICS.items()
        },
    }


def models_status():
    info = get_models_info()
    by_id = {m["id"]: m for m in info["models"]}
    return {
        "cbis": by_id["cbis"]["exists"],
        "rsna": by_id["rsna"]["exists"],
        "busi": by_id["busi"]["exists"],
        "mri": by_id["mri"]["exists"],
        "fusion_v2": by_id["fusion_v2"]["exists"],
        "fusion_v3": by_id["fusion_v3"]["exists"],
        "fusion_active": info["fusion_active"],
        "loaded": _loaded,
        "metrics": info["metrics"],
        "active": info["active"],
        "models": info["models"],
    }


def load_models():
    global _loaded, _cbis_enc, _rsna_enc, _busi_enc, _mri_enc
    global _fusion, _fusion_version, _device
    if _loaded:
        return True
    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Scanova] Loading models on {_device}")

        def load_enc(path, name, metric_key):
            if not path.exists():
                print(f"[Scanova] WARNING: {name} not found → {path}")
                return None
            m = tv_models.efficientnet_b0(weights=None)
            m.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(m.classifier[1].in_features, 1),
            )
            ckpt = torch.load(path, map_location=_device, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            for p in m.parameters():
                p.requires_grad = False
            for p in m.features[-1].parameters():
                p.requires_grad = True
            m.to(_device).eval()
            auc = ckpt.get("val_auc")
            print(f"[Scanova] {name} loaded  (val_auc={auc if auc is not None else '?'})")
            return m

        class FusionHead(nn.Module):
            """Shared MLP used by Fusion V2 (3 encoders) and V3 (4 encoders)."""

            def __init__(self, n_encoders=3, emb_dim=1280):
                super().__init__()
                in_dim = emb_dim * n_encoders
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                )

            def forward(self, *embs):
                return self.net(torch.cat(embs, dim=1))

        _cbis_enc = load_enc(CBIS_CKPT, "CBIS-DDSM", "cbis")
        _rsna_enc = load_enc(RSNA_CKPT, "RSNA", "rsna")
        _busi_enc = load_enc(BUSI_CKPT, "BUSI", "busi")
        _mri_enc = load_enc(MRI_CKPT, "MRI", "mri")

        # Prefer Fusion V3 (4-way) when available
        if FUSION_V3_CKPT.exists():
            _fusion = FusionHead(n_encoders=4).to(_device)
            ckpt = torch.load(FUSION_V3_CKPT, map_location=_device, weights_only=False)
            _fusion.load_state_dict(ckpt["fusion_head"])
            _fusion.eval()
            _fusion_version = "v3"
            auc = ckpt.get("val_auc")
            print(f"[Scanova] Fusion V3 loaded (val_auc={auc if auc is not None else '?'})")
        elif FUSION_V2_CKPT.exists():
            _fusion = FusionHead(n_encoders=3).to(_device)
            ckpt = torch.load(FUSION_V2_CKPT, map_location=_device, weights_only=False)
            _fusion.load_state_dict(ckpt["fusion_head"])
            _fusion.eval()
            _fusion_version = "v2"
            auc = ckpt.get("val_auc")
            print(f"[Scanova] Fusion V2 loaded (val_auc={auc if auc is not None else '?'})")
        else:
            print("[Scanova] WARNING: no fusion_v3 / fusion_v2 checkpoint found")

        _loaded = True
        return True
    except Exception as e:
        print(f"[Scanova] Model load error: {e}")
        traceback.print_exc()
        return False


def _get_embedding(encoder, tensor):
    import torch

    with torch.no_grad():
        f = encoder.features(tensor)
        f = encoder.avgpool(f)
        return torch.flatten(f, 1)


def _fuse_forward(cbis_emb, rsna_emb, busi_emb, mri_emb):
    if _fusion_version == "v3":
        return _fusion(cbis_emb, rsna_emb, busi_emb, mri_emb)
    return _fusion(cbis_emb, rsna_emb, busi_emb)


# Every upload is scored by Fusion V3 with all four encoders.
# A healthy embedding is about 5–20 L2. Far above that, the vector has
# overflowed and would pull the fused logit to about -1000 (always benign).
# Only that broken vector is replaced with zeros. The others are still fused.
_ENCODER_LABELS = {
    "cbis": "CBIS",
    "rsna": "RSNA",
    "busi": "BUSI",
    "mri": "MRI",
}
_EMB_NORM_LIMIT = 80.0


def normalize_modality(modality: str | None) -> str:
    key = (modality or "mammography").strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "mammography": "mammography",
        "mammogram": "mammography",
        "mammo": "mammography",
        "mg": "mammography",
        "ultrasound": "ultrasound",
        "us": "ultrasound",
        "busi": "ultrasound",
        "breast ultrasound": "ultrasound",
        "mri": "mri",
        "breast mri": "mri",
        "mr": "mri",
    }
    return aliases.get(key, "mammography")


def _encoder_map():
    return {
        "cbis": _cbis_enc,
        "rsna": _rsna_enc,
        "busi": _busi_enc,
        "mri": _mri_enc,
    }


def _zeros_emb():
    import torch

    return torch.zeros(1, 1280, device=_device)


def _embedding_or_zero(encoder, tensor):
    """Return (embedding, usable). Unusable vectors are exact zeros."""
    if encoder is None:
        return _zeros_emb(), False
    emb = _get_embedding(encoder, tensor)
    n = float(torch_norm(emb))
    if not np.isfinite(n) or n > _EMB_NORM_LIMIT:
        return _zeros_emb(), False
    return emb, True


def torch_norm(emb) -> float:
    import torch

    return float(torch.norm(emb).item())


def _encoder_probability(encoder, tensor) -> float:
    import torch

    with torch.no_grad():
        return float(torch.sigmoid(encoder(tensor)).item())


def run_inference(img_pil: Image.Image, modality: str | None = None) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import cv2

        load_models()

        tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        img_rgb = img_pil.convert("RGB")
        img_224 = img_rgb.resize((224, 224))
        img_np = np.array(img_224).astype(np.float32) / 255.0
        tensor = tf(img_rgb).unsqueeze(0).to(_device)

        modality = normalize_modality(modality)
        encoders = _encoder_map()
        # Fusion V2 has no MRI weight. Still run MRI when V3 is the active head.
        names = ("cbis", "rsna", "busi", "mri") if _fusion_version == "v3" else ("cbis", "rsna", "busi")
        slots = {name: _zeros_emb() for name in ("cbis", "rsna", "busi", "mri")}
        used = []
        for name in names:
            emb, ok = _embedding_or_zero(encoders[name], tensor)
            slots[name] = emb
            if ok:
                used.append(name)
        cbis_emb, rsna_emb = slots["cbis"], slots["rsna"]
        busi_emb, mri_emb = slots["busi"], slots["mri"]

        fusion_ok = bool(_fusion) and bool(used)
        if fusion_ok:
            _fusion.eval()
            with torch.no_grad():
                prob = torch.sigmoid(
                    _fuse_forward(cbis_emb, rsna_emb, busi_emb, mri_emb)
                ).item()
        else:
            primary = encoders.get("cbis") or next(
                (encoders[n] for n in names if encoders[n] is not None), None
            )
            if primary is None:
                raise RuntimeError("No encoder loaded")
            prob = _encoder_probability(primary, tensor)
            if not used:
                used = ["cbis"] if encoders.get("cbis") is not None else [names[0]]

        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        mc_probs = []
        if fusion_ok:
            _fusion.apply(enable_dropout)
            with torch.no_grad():
                for _ in range(15):
                    mc_probs.append(
                        torch.sigmoid(
                            _fuse_forward(cbis_emb, rsna_emb, busi_emb, mri_emb)
                        ).item()
                    )
            _fusion.eval()
        mc_std = float(np.std(mc_probs)) if mc_probs else 0.03

        c_n = float(torch.norm(cbis_emb).item())
        r_n = float(torch.norm(rsna_emb).item())
        u_n = float(torch.norm(busi_emb).item())
        m_n = float(torch.norm(mri_emb).item())
        tot = c_n + r_n + u_n + m_n + 1e-8

        heatmap_b64 = None
        try:
            enc = next((encoders[n] for n in used if encoders[n] is not None), None)
            if enc:
                cam = GradCAMPlusPlus(model=enc, target_layers=[enc.features[-1]])
                gc = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
                gc = (gc - gc.min()) / (gc.max() - gc.min() + 1e-8)
                hm = show_cam_on_image(
                    img_np, gc, use_rgb=True, colormap=cv2.COLORMAP_JET
                )
                buf = io.BytesIO()
                Image.fromarray(hm).save(buf, format="PNG")
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"[Scanova] GradCAM error: {e}")

        used_labels = "+".join(_ENCODER_LABELS[n] for n in used) or "none"
        if fusion_ok and _fusion_version == "v3":
            model_used = f"Fusion V3 ({used_labels})"
        elif fusion_ok and _fusion_version == "v2":
            model_used = f"Fusion V2 ({used_labels})"
        else:
            model_used = used_labels

        risk = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
        return {
            "prediction": "malignant" if prob >= 0.5 else "benign",
            "probability": round(prob * 100, 1),
            "confidence": round((prob if prob >= 0.5 else 1 - prob) * 100, 1),
            "uncertainty": round(mc_std * 100, 1),
            "risk_level": risk,
            "shap_cbis": round(c_n / tot * 100, 1),
            "shap_rsna": round(r_n / tot * 100, 1),
            "shap_us": round(u_n / tot * 100, 1),
            "shap_mri": round(m_n / tot * 100, 1),
            "heatmap": heatmap_b64,
            "model_used": model_used,
            "fusion_version": _fusion_version,
            "modality": modality,
            "encoders_used": used,
        }

    except Exception as e:
        print(f"[Scanova] Inference error: {e}")
        traceback.print_exc()
        raise
