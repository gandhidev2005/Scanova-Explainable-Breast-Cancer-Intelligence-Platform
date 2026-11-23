"""External (VinDr) and internal test-set validation figures for dashboards.

Figures match the paper comparison table. VinDr-Mammo is research /
domain-shift validation only — not clinical performance.
"""

EXTERNAL_VALIDATION = {
    "dataset": "VinDr-Mammo",
    "display_name": "Fusion V3 VinDr (External Validation)",
    "origin": "Ho Chi Minh City, Vietnam",
    "hospital_type": "Two major Vietnamese hospitals",
    "n_images": 398,
    "positive": 198,
    "negative": 200,
    "excluded": "BI-RADS 3 (ambiguous — not used)",
    "models_tested": [
        "Fusion V2 VinDr (External Validation)",
        "Fusion V3 VinDr (External Validation)",
    ],
    "retraining": "None — models tested as-is",
    "methodology": "No retraining. Threshold from internal val set.",
    "domain_shift_note": "Expected. Different scanners and patient population.",
    "threshold_source": "CBIS-DDSM internal val set (frozen)",
    "modalities": "Mammography only (ultrasound + MRI zeroed at inference)",
    "fusion_v2": {
        "auc": 0.6319,
        "sensitivity": 0.3586,
        "specificity": 0.7950,
        "accuracy": 57.8,
    },
    "fusion_v3": {
        "auc": 0.6549,
        "sensitivity": 0.5657,
        "specificity": 0.6950,
        "accuracy": 63.1,
    },
    "internal": {
        "fusion_v2": {
            "auc": 0.9913,
            "sensitivity": 0.9659,
            "specificity": 0.9948,
            "accuracy": 98.3,
        },
        "fusion_v3": {
            "auc": 0.9897,
            "sensitivity": 0.9697,
            "specificity": 1.0000,
            "accuracy": 98.8,
        },
    },
    "auc_drop_v2": 0.3594,
    "auc_drop_v3": 0.3348,
    "v3_generalisation_gain": 0.023,
}
