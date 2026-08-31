╔══════════════════════════════════════════════════════════════╗
║                Scanova — Breast Cancer AI Platform          ║
║         Flask + Modular Backend + Single-Page App (SPA)     ║
╚══════════════════════════════════════════════════════════════╝

Scanova is a clinical decision-support web platform for breast-cancer
screening. It combines four deep-learning models (three modality encoders
plus a fusion head), Grad-CAM++ explainability, Monte Carlo uncertainty
estimation, and a Groq-powered Explainable AI chat assistant — all served
from one Flask backend and one HTML frontend.

Stack:  Python 3.10+ (tested on 3.14.6) · Flask · JSON storage · Vanilla JS SPA
AI:     EfficientNet-B0 ×3 + FusionHeadV2 · Grad-CAM++ · MC Dropout (×15)
Chat:   Groq LLaMA 3.3 70B (Explainable AI + general assistant)


═══════════════════════════════════════════════════
TABLE OF CONTENTS
═══════════════════════════════════════════════════

  1.  What the website does (overview)
  2.  User roles & dashboards
  3.  AI models — architecture & accuracy
  4.  Inference pipeline (step-by-step)
  5.  Explainable AI (XAI) & chat
  6.  Folder structure
  7.  Setup & run
  8.  Environment variables (.env)
  9.  Security
  10. Data storage (JSON + future MySQL)
  11. All features & UI pages
  12. API endpoints
  13. Troubleshooting


═══════════════════════════════════════════════════
1. WHAT THE WEBSITE DOES
═══════════════════════════════════════════════════

Scanova helps patients, radiologists, clinic staff, and administrators
work with AI-assisted breast imaging analysis:

  • Upload mammograms or ultrasound images (JPG, PNG, DICOM)
  • Run Fusion V2 ensemble inference with heatmaps and risk scoring
  • Chat with an AI assistant that explains results in plain language
  • Manage patients, appointments, scan history, and user accounts
  • Request and approve professional role upgrades (patient → radiologist/clinic)

The site is a Single-Page Application (frontend/index.html) served by Flask.
All API calls go to /api/* endpoints. Login uses JWT tokens stored in the
browser; sessions persist across page refresh.


═══════════════════════════════════════════════════
2. USER ROLES & DASHBOARDS
═══════════════════════════════════════════════════

  PATIENT (default on registration)
  ─────────────────────────────────
  • Upload scans via Diagnostic Lens
  • View personal scan history and risk levels
  • Explainable AI chat (simple, non-technical language)
  • Direct AI Assistant for general breast-health questions
  • Request role upgrade (radiologist / clinic / admin)

  RADIOLOGIST
  ───────────
  • All patient scan features for own uploads
  • Scheduling: view/add/update/delete appointments
  • Dashboard: pending queue, throughput chart, critical findings
  • AI chat in clinical / BI-RADS-oriented language
  • View patient registry (read-only)

  CLINIC
  ──────
  • Patient registry: add, view, delete patients
  • Analytics dashboard with model performance reference
  • AI chat focused on triage and workflow next steps
  • Scheduling access

  ADMIN
  ─────
  • Full user management (view, delete, change roles)
  • User detail page: scans, role requests, chat stats, PDF export
  • Role request approval/rejection
  • Admin audit logs
  • System dashboard: user counts, model status, pending requests
  • AI chat with selectable response style:
      normal (patient) · radio (radiologist) · clinic (clinic staff)

  PRE-SEEDED ADMIN (only built-in account)
  ────────────────────────────────────────
  Email:    set via ADMIN_EMAIL in .env (default: admin@scanova.com)
  Password: set via ADMIN_PASSWORD in .env (use a strong unique password)
  (See Sections 8–9 for security configuration)


═══════════════════════════════════════════════════
3. AI MODELS — ARCHITECTURE & ACCURACY
═══════════════════════════════════════════════════

Scanova uses FOUR PyTorch checkpoint files in the models/ folder.
Three are EfficientNet-B0 encoders trained on different imaging datasets;
the fourth is FusionHeadV2, which combines all three embeddings into
one malignancy probability.

┌─────────────────────────────────────────────────────────────────────┐
│                        FUSION V2 PIPELINE                           │
│                                                                     │
│   Input image (224×224 RGB)                                         │
│        │                                                            │
│        ├──► CBIS-DDSM Encoder  ──► 1280-dim embedding ──┐         │
│        ├──► RSNA Encoder       ──► 1280-dim embedding ──┼──► Fuse  │
│        └──► BUSI-US Encoder    ──► 1280-dim embedding ──┘    │      │
│                                                               ▼      │
│                    FusionHeadV2 (3840 → 1024 → 256 → 64 → 1)        │
│                               sigmoid → probability                  │
│                               MC Dropout ×15 → uncertainty           │
│                               L2 norm ratio → modality %             │
│                               Grad-CAM++ → heatmap overlay           │
└─────────────────────────────────────────────────────────────────────┘


MODEL FILES
───────────

  File                    Role                         Dataset
  ──────────────────────  ───────────────────────────  ──────────────────
  best_model.pth          CBIS-DDSM mammography encoder CBIS-DDSM
  rsna_model.pth          RSNA mammography encoder      RSNA (mammo)
  busi_model.pth          Ultrasound encoder            BUSI-US
  fusion_v2_model.pth     Fusion head (ensemble)        Combined (all 3)


INDIVIDUAL ENCODER PERFORMANCE (validation / test sets)
──────────────────────────────────────────────────────

  Model                   AUC-ROC   Sensitivity   Specificity   Accuracy
  ──────────────────────  ────────  ────────────  ────────────  ────────
  Mammography (Set A)     0.7434    71.97%        63.25%        66.8%
    CBIS-DDSM encoder     best_model.pth

  Mammography (Set B)     0.6817    61.49%        67.77%        66.0%
    RSNA encoder            rsna_model.pth

  Ultrasound              0.9647    83.87%        96.51%        93.2%
    BUSI-US encoder         busi_model.pth


FUSION V2 ENSEMBLE ★ (production inference model)
──────────────────────────────────────────────────

  Metric              Value       Notes
  ──────────────────  ──────────  ─────────────────────────────────────
  AUC-ROC             0.9913      Primary ranking metric (test set)
  Accuracy            98.3%       Overall correct classifications
  Sensitivity         96.59%      True positive rate (malignant detected)
  Specificity         99.48%      True negative rate (benign correctly ruled out)
  Inference time      < 2 s       CPU/GPU dependent; typical on modern hardware

  Fusion V2 outperforms every individual encoder by combining multi-modal
  signals. This is the model used for live predictions when all four .pth
  files are present. If only CBIS is available, the app falls back to
  single-encoder mode; if no models load, demo-mode defaults are returned.


ARCHITECTURE DETAILS
────────────────────

  Backbone:       EfficientNet-B0 (Tan & Le, 2019) × 3 encoders
  Embedding dim:  1280 per encoder (3840 concatenated)
  Fusion head:    FusionHeadV2 — Linear(3840→1024) → ReLU → Dropout(0.4)
                  → Linear(1024→256) → ReLU → Dropout(0.3)
                  → Linear(256→64)  → ReLU → Dropout(0.2)
                  → Linear(64→1)    → sigmoid
  Input size:     224 × 224 pixels, ImageNet normalization
  Threshold:      ≥ 0.5 → malignant; < 0.5 → benign
  Risk levels:    low (< 40%) · medium (40–70%) · high (> 70%)


EXPLAINABILITY & UNCERTAINTY METHODS
────────────────────────────────────

  Method              Purpose
  ──────────────────  ───────────────────────────────────────────────────
  Grad-CAM++          Visual heatmap — highlights tissue regions that
                      drove the prediction (Selvaraju et al. 2020)
  MC Dropout (×15)    Epistemic uncertainty estimate (Gal & Ghahramani 2016)
  L2 norm ratio       Modality contribution % (CBIS / RSNA / BUSI signal strength)
  Groq LLaMA 3.3 70B  Natural-language explanation of scan metrics


═══════════════════════════════════════════════════
4. INFERENCE PIPELINE (STEP-BY-STEP)
═══════════════════════════════════════════════════

  1. User uploads PNG / JPG / DICOM via Diagnostic Lens
  2. Browser → POST /api/scan/analyze (multipart form, JWT auth)
  3. Backend loads image:
       • DICOM: pydicom pixel array + CLAHE contrast enhancement
       • JPG/PNG: Pillow RGB conversion
       • AVIF: rejected with clear error message
  4. Models lazy-loaded from models/ on first inference request
  5. Image resized to 224×224, ImageNet-normalized tensor
  6. Three EfficientNet-B0 encoders extract 1280-dim embeddings each
  7. FusionHeadV2 concatenates [3840] → sigmoid → malignancy probability
  8. Monte Carlo Dropout: 15 forward passes → std dev as uncertainty %
  9. L2 norm of each embedding → modality contribution % (shown as SHAP bars)
  10. Grad-CAM++ on CBIS (or RSNA fallback) encoder → heatmap PNG (base64)
  11. Result saved to data/scans.json → JSON response to frontend
  12. Frontend displays prediction, confidence, risk badge, heatmap, XAI chat


═══════════════════════════════════════════════════
5. EXPLAINABLE AI (XAI) & CHAT
═══════════════════════════════════════════════════

Scanova has TWO AI chat systems powered by Groq (LLaMA 3.3 70B):

  A) Scan-specific chat  —  POST /api/scan/chat
     Triggered after a scan analysis. The AI receives:
       prediction, probability, confidence, uncertainty, risk level,
       modality contribution %, model name, filename
     Response style adapts to user role:
       • Patient: plain language, reassuring tone
       • Radiologist: BI-RADS, clinical terminology, guidelines
       • Clinic: triage urgency, workflow next steps
     Chat history saved per scan in data/scan_chats.json

  B) General / Direct chat  —  POST /api/chat/general
                               POST /api/direct-chats/<id>/message
     ChatGPT-style assistant for breast-health questions without a scan.
     Session history stored in data/direct_chat_sessions.json and
     data/direct_chat_messages.json. Admin can switch response mode
     (normal / radiologist / clinic) with isolated session histories.

  Configure Groq in .env:  GROQ_API_KEY=your-key-here


═══════════════════════════════════════════════════
6. FOLDER STRUCTURE
═══════════════════════════════════════════════════

  Scanova/
  ├── app.py                      ← Main entry point — run this
  ├── requirements.txt            ← Python dependencies
  ├── .env                        ← Local environment config (not committed)
  ├── .env.example                ← Template for .env
  │
  ├── data/                       ← JSON storage (active runtime database)
  │   ├── users.json
  │   ├── scans.json
  │   ├── patients.json
  │   ├── appointments.json
  │   ├── role_requests.json
  │   ├── admin_logs.json
  │   ├── scan_chats.json
  │   ├── direct_chat_sessions.json
  │   └── direct_chat_messages.json
  │
  ├── database/
  │   └── mysql_schema.sql        ← Future MySQL schema (not active yet)
  │
  ├── backend/
  │   ├── config.py               ← Loads .env settings
  │   ├── database.py             ← Compatibility shim
  │   ├── storage/
  │   │   └── json_store.py       ← JSON read/write layer
  │   ├── middleware/
  │   │   └── auth.py             ← JWT helpers, require_auth / require_role
  │   ├── services/
  │   │   ├── groq_service.py     ← Groq API calls
  │   │   ├── prompt_service.py   ← XAI system prompts by role
  │   │   ├── image_service.py    ← JPG/PNG/DICOM loading
  │   │   └── admin_log_service.py
  │   ├── inference/
  │   │   └── engine.py           ← Model loading + run_inference()
  │   └── routes/                 ← One file per domain
  │       ├── auth_routes.py
  │       ├── mfa_routes.py
  │       ├── scan_analyze_routes.py
  │       ├── scan_chat_routes.py
  │       ├── general_chat_routes.py
  │       ├── direct_chat_routes.py
  │       ├── scans_routes.py
  │       ├── patients_routes.py
  │       ├── schedule_routes.py
  │       ├── dashboard_routes.py
  │       ├── users_routes.py
  │       ├── role_requests_routes.py
  │       ├── admin_logs_routes.py
  │       └── health_routes.py
  │
  ├── models/                     ← Place all 4 .pth files HERE
  │   ├── best_model.pth          ← CBIS-DDSM encoder
  │   ├── rsna_model.pth          ← RSNA encoder
  │   ├── busi_model.pth          ← BUSI-US encoder
  │   └── fusion_v2_model.pth     ← FusionHeadV2
  │
  └── frontend/
      └── index.html              ← Complete SPA (HTML + CSS + JS)


═══════════════════════════════════════════════════
7. SETUP & RUN
═══════════════════════════════════════════════════

STEP 1 — Place model files
──────────────────────────
  Copy all 4 .pth files into models/:
    best_model.pth · rsna_model.pth · busi_model.pth · fusion_v2_model.pth

STEP 2 — Configure environment
──────────────────────────────
  copy .env.example .env
  Edit .env — set GROQ_API_KEY and SCANOVA_SECRET at minimum.

STEP 3 — Install dependencies
─────────────────────────────
  python -m venv .venv
  .venv\Scripts\activate          (Windows)
  pip install -r requirements.txt

  PyTorch (included in requirements.txt for CPU).
  For NVIDIA GPU, reinstall torch after requirements:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

STEP 4 — Run
────────────
  python app.py

  Expected console output:
  ============================================================
    Scanova - Breast Cancer AI
  ============================================================
    CBIS model  : [OK] / [MISSING - copy to models/]
    RSNA model  : [OK] / [MISSING - copy to models/]
    BUSI model  : [OK] / [MISSING - copy to models/]
    Fusion V2   : [OK] / [MISSING - copy to models/]
  ============================================================
    Storage     : JSON (data)
    Admin email : (from .env ADMIN_EMAIL)
    URL         : http://localhost:5000
  ============================================================
  Security notices may appear if .env is not fully configured.

STEP 5 — Open browser
─────────────────────
  http://localhost:5000
  Health check: http://localhost:5000/api/health


═══════════════════════════════════════════════════
8. ENVIRONMENT VARIABLES (.env)
═══════════════════════════════════════════════════

  All secrets live in .env ONLY — never in source code.
  Copy .env.example → .env and fill in values before running.

  Variable                Purpose
  ──────────────────────  ──────────────────────────────────────────
  SCANOVA_SECRET          JWT signing key (≥32 random chars)
  GROQ_API_KEY            Groq API key for AI chat (from console.groq.com)
  ADMIN_EMAIL             Seed admin email
  ADMIN_PASSWORD          Seed admin password (strong, unique)
  CORS_ORIGINS            Allowed browser origins (comma-separated)
  MAX_UPLOAD_MB           Max scan upload size (default 16)
  MIN_PASSWORD_LENGTH     Min password length (default 8)
  RATE_LIMIT_LOGIN        Max login attempts per IP (default 10)
  RATE_LIMIT_WINDOW_SEC   Rate limit window (default 900 = 15 min)
  ALLOW_INSECURE_DEFAULTS true = dev only; must be false in production

  Generate a strong JWT secret:
    python -c "import secrets; print(secrets.token_urlsafe(48))"


═══════════════════════════════════════════════════
9. SECURITY
═══════════════════════════════════════════════════

  • No API keys or passwords in source code — .env only (.gitignored)
  • bcrypt password hashing (passlib + bcrypt 4.0.1)
  • JWT authentication on all protected /api/* routes
  • Role-based access control (patient / radiologist / clinic / admin)
  • Optional TOTP MFA (Google Authenticator compatible)
  • Login rate limiting (10 attempts / 15 min per IP)
  • Password policy: ≥8 chars, at least one letter and one number
  • CORS restricted to configured origins (not open *)
  • Security headers: X-Frame-Options, nosniff, HSTS, Referrer-Policy
  • Upload limits: max file size, extension whitelist (jpg/png/dcm)
  • Sanitized filenames (basename only, no path traversal)
  • Generic error messages — no stack traces or API keys in responses
  • Sensitive fields (password, mfa_secret) stripped from API responses
  • Startup security audit prints warnings for weak/missing config


═══════════════════════════════════════════════════
10. DATA STORAGE
═══════════════════════════════════════════════════

  ACTIVE:   JSON files in data/ (one file per table)
  FUTURE:   MySQL — schema ready at database/mysql_schema.sql
            Switch when STORAGE_BACKEND=mysql is implemented.

  Tables stored as JSON:
    users · scans · patients · appointments · role_requests
    admin_logs · scan_chats · direct_chat_sessions · direct_chat_messages

  Admin account is auto-seeded on first run from .env credentials.
  All other users register via the UI and start as role: patient.


═══════════════════════════════════════════════════
11. ALL FEATURES & UI PAGES
═══════════════════════════════════════════════════

LANDING PAGE
  ✓ Hero with Fusion V2 stats (AUC 0.9913 · 98.3% accuracy)
  ✓ Sign In / Register · smooth-scroll nav (About / Diagnostics / Contact)
  ✓ Model performance bar chart (all 4 models with AUC / sens / spec / acc)
  ✓ XAI cards (Grad-CAM++, MC Dropout, modality attribution)
  ✓ Breast Cancer Awareness section with symptom cards
  ✓ Privacy / Terms / HIPAA / Contact modals

LOGIN & REGISTER
  ✓ JWT login with optional MFA (TOTP)
  ✓ Password show/hide · forgot-password modal (contact admin)
  ✓ Self-registration → default role: patient

DIAGNOSTIC LENS (scan upload)
  ✓ Drag-and-drop + multi-file upload (JPG / PNG / DICOM)
  ✓ Fusion V2 inference per file
  ✓ Results: prediction · probability · confidence · uncertainty
  ✓ Risk level badge (low / medium / high)
  ✓ Modality contribution bars (CBIS / RSNA / BUSI-US)
  ✓ Grad-CAM++ heatmap overlay
  ✓ Patient-facing Riskometer gauge
  ✓ Batch results grid for multi-file uploads
  ✓ AVIF rejected with clear on-screen error

SCAN DETAIL
  ✓ Full-page view with heatmap + all metrics
  ✓ Explainable AI chat integrated (history saved per scan)
  ✓ Rename scan label · delete scan

AI ASSISTANT (direct chat)
  ✓ ChatGPT-style layout with session sidebar
  ✓ New / rename / delete chat sessions
  ✓ Role-aware responses (patient / radiologist / clinic / admin modes)
  ✓ Symptom triage with actionable guidance

SCHEDULING (radiologist)
  ✓ View / add / update status / delete appointments
  ✓ Today's queue highlighted · history for completed/cancelled

PATIENT RECORDS (clinic / admin)
  ✓ Add / view / delete patients with risk badges

USER MANAGEMENT (admin)
  ✓ View all users · delete · change roles
  ✓ User detail: scans + role requests + stats + PDF export

ROLE REQUESTS
  ✓ Patients request upgrade · admin approve/reject · pending count badge

SCAN HISTORY
  ✓ All personal scans with prediction + risk · click for full detail

ANALYTICS (clinic)
  ✓ Live stats + model performance reference table (all 4 models)

SETTINGS
  ✓ Update name / email · change password · MFA setup/disable · sign out

ADMIN DASHBOARD
  ✓ System stats · model load status · export report · audit logs


═══════════════════════════════════════════════════
12. API ENDPOINTS
═══════════════════════════════════════════════════

  AUTH
  POST   /api/auth/login              Login → JWT token
  POST   /api/auth/register           Register (role: patient)
  GET    /api/auth/me                 Current user
  PUT    /api/auth/profile            Update name + email
  PUT    /api/auth/password           Change password
  GET    /api/auth/mfa/status         MFA enabled?
  POST   /api/auth/mfa/start          Start MFA (returns QR + secret)
  POST   /api/auth/mfa/verify         Verify TOTP + enable MFA
  POST   /api/auth/mfa/disable        Disable MFA (password + code required)

  SCAN & AI
  POST   /api/scan/analyze            Upload image → Fusion V2 inference
  POST   /api/scan/chat               Scan-specific XAI chat
  POST   /api/chat/general            General AI chat (no scan context)

  SCANS
  GET    /api/scans                   Own scan history
  GET    /api/scans/<id>              Single scan (admin: any; others: own)
  PATCH  /api/scans/<id>/label        Rename scan label
  DELETE /api/scans/<id>              Delete scan
  GET    /api/scans/<id>/chats        Scan chat history
  DELETE /api/scans/<id>/chats        Clear scan chat history

  DIRECT CHAT
  GET    /api/direct-chats            List sessions
  GET    /api/direct-chats?style=     Filter by mode (admin)
  POST   /api/direct-chats            Create session
  GET    /api/direct-chats/<sid>      Session + messages
  POST   /api/direct-chats/<sid>/message   Send message
  PATCH  /api/direct-chats/<sid>/title     Rename session
  DELETE /api/direct-chats/<sid>           Delete session

  DASHBOARD & CLINIC
  GET    /api/dashboard/stats         Role-specific dashboard data
  GET    /api/patients                Patient list
  POST   /api/patients                Add patient
  DELETE /api/patients/<id>           Remove patient
  GET    /api/schedule                List appointments
  POST   /api/schedule                Add appointment
  PUT    /api/schedule/<id>           Update appointment status
  DELETE /api/schedule/<id>           Delete appointment

  ADMIN
  GET    /api/users                   All users
  DELETE /api/users/<id>              Delete user
  PUT    /api/users/<id>/role         Change role
  GET    /api/users/<id>/detail       Full user profile + scans + stats
  GET    /api/admin-logs              Audit trail
  POST   /api/role-requests           Submit role upgrade request
  GET    /api/role-requests/my        Own requests
  GET    /api/role-requests           All requests (admin)
  PUT    /api/role-requests/<id>      Approve / reject (admin)
  GET    /api/role-requests/pending-count

  SYSTEM
  GET    /api/health                  Model file + load status


═══════════════════════════════════════════════════
13. TROUBLESHOOTING
═══════════════════════════════════════════════════

  "AVIF images are not supported"
    → Convert to JPG or PNG before uploading.

  UI looks stale after code changes
    → Restart Flask (Ctrl+C → python app.py)
    → Hard refresh browser: Ctrl+Shift+R

  AI chat returns 502 / unavailable
    → Check GROQ_API_KEY in .env
    → Verify internet connectivity to api.groq.com

  Models show [MISSING]
    → Copy all 4 .pth files into models/
    → Restart the server

  Login fails after restructure
    → JSON storage starts fresh; admin is re-seeded from .env
    → Register new accounts for other users

  bcrypt / passlib warnings
    → requirements.txt pins bcrypt==4.0.1 for passlib compatibility


═══════════════════════════════════════════════════
REFERENCES
═══════════════════════════════════════════════════

  EfficientNet:     Tan & Le, 2019
  Grad-CAM++:       Selvaraju et al., 2020
  MC Dropout:       Gal & Ghahramani, 2016
  Datasets:         CBIS-DDSM · RSNA Mammography · BUSI Ultrasound

  Scanova is a research / decision-support prototype.
  It does not replace professional medical diagnosis.
