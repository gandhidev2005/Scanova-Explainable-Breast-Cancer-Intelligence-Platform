-- Scanova MySQL schema (future production storage)
-- Active runtime storage is JSON (see backend/storage/json_store.py).
-- Apply when STORAGE_BACKEND=mysql is implemented.

CREATE DATABASE IF NOT EXISTS scanova
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE scanova;

-- ── Users & authentication ───────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(120) NOT NULL,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          ENUM('patient','radiologist','clinic','admin') NOT NULL DEFAULT 'patient',
    mfa_enabled   TINYINT(1) NOT NULL DEFAULT 0,
    mfa_secret    VARCHAR(64) NULL,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_users_role (role),
    INDEX idx_users_created (created_at)
) ENGINE=InnoDB;

-- ── AI scan results ────────────────────────────────────
CREATE TABLE IF NOT EXISTS scans (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    uploaded_by   INT UNSIGNED NOT NULL,
    filename      VARCHAR(512) NULL,
    label         VARCHAR(120) NULL,
    modality      VARCHAR(64) NOT NULL DEFAULT 'mammography',
    prediction    ENUM('malignant','benign') NULL,
    probability   DECIMAL(5,2) NULL,
    confidence    DECIMAL(5,2) NULL,
    uncertainty   DECIMAL(5,2) NULL,
    risk_level    ENUM('low','medium','high') NULL,
    heatmap       LONGTEXT NULL,
    shap_mammo    DECIMAL(5,2) NULL,
    shap_rsna     DECIMAL(5,2) NULL,
    shap_us       DECIMAL(5,2) NULL,
    model_used    VARCHAR(120) NULL,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_scans_user FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_scans_user (uploaded_by),
    INDEX idx_scans_risk (risk_level),
    INDEX idx_scans_created (created_at)
) ENGINE=InnoDB;

-- ── Clinic patient registry ────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id          INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(120) NOT NULL,
    dob         DATE NULL,
    email       VARCHAR(255) NULL,
    phone       VARCHAR(32) NULL,
    risk_level  ENUM('low','medium','high') NOT NULL DEFAULT 'low',
    last_scan   DATETIME NULL,
    notes       TEXT NULL,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_patients_risk (risk_level)
) ENGINE=InnoDB;

-- ── Appointments / schedule ────────────────────────────
CREATE TABLE IF NOT EXISTS appointments (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    patient_name  VARCHAR(120) NOT NULL,
    appt_date     DATE NOT NULL,
    appt_time     TIME NOT NULL,
    type          VARCHAR(64) NOT NULL DEFAULT 'Mammography',
    doctor        VARCHAR(120) NULL,
    status        ENUM('scheduled','in_progress','completed','cancelled') NOT NULL DEFAULT 'scheduled',
    notes         TEXT NULL,
    created_by    INT UNSIGNED NULL,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_appt_creator FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_appt_date (appt_date, appt_time),
    INDEX idx_appt_status (status)
) ENGINE=InnoDB;

-- ── Role upgrade workflow ──────────────────────────────
CREATE TABLE IF NOT EXISTS role_requests (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id         INT UNSIGNED NOT NULL,
    current_role    VARCHAR(32) NOT NULL,
    requested_role  VARCHAR(32) NOT NULL,
    reason          TEXT NULL,
    status          ENUM('pending','approved','rejected') NOT NULL DEFAULT 'pending',
    admin_notes     TEXT NULL,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at     DATETIME NULL,
    resolved_by     INT UNSIGNED NULL,
    CONSTRAINT fk_rr_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_rr_admin FOREIGN KEY (resolved_by) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_rr_status (status),
    INDEX idx_rr_user (user_id)
) ENGINE=InnoDB;

-- ── Admin audit trail ──────────────────────────────────
CREATE TABLE IF NOT EXISTS admin_logs (
    id           INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    admin_id     INT UNSIGNED NULL,
    admin_name   VARCHAR(120) NOT NULL,
    action       VARCHAR(120) NOT NULL,
    target_id    INT UNSIGNED NULL,
    target_name  VARCHAR(120) NULL,
    details      TEXT NULL,
    created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_log_admin FOREIGN KEY (admin_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_logs_created (created_at)
) ENGINE=InnoDB;

-- ── Explainable AI chat (per scan) ─────────────────────
CREATE TABLE IF NOT EXISTS scan_chats (
    id         INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    scan_id    INT UNSIGNED NOT NULL,
    user_id    INT UNSIGNED NOT NULL,
    role       ENUM('user','assistant') NOT NULL,
    content    TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_schat_scan FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
    CONSTRAINT fk_schat_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_schat_scan (scan_id)
) ENGINE=InnoDB;

-- ── Direct AI chat sessions ────────────────────────────
CREATE TABLE IF NOT EXISTS direct_chat_sessions (
    id         INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id    INT UNSIGNED NOT NULL,
    title      VARCHAR(120) NOT NULL DEFAULT 'New Chat',
    style      ENUM('normal','radio','clinic') NOT NULL DEFAULT 'normal',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_dcs_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_dcs_user (user_id),
    INDEX idx_dcs_updated (updated_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS direct_chat_messages (
    id         INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id INT UNSIGNED NOT NULL,
    role       ENUM('user','assistant') NOT NULL,
    content    TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_dcm_session FOREIGN KEY (session_id) REFERENCES direct_chat_sessions(id) ON DELETE CASCADE,
    INDEX idx_dcm_session (session_id)
) ENGINE=InnoDB;

-- ── Seed admin (optional — prefer env-driven seed in app) ─
-- INSERT INTO users (name, email, password_hash, role)
-- VALUES ('Administrator', 'admin@scanova.com', '<bcrypt-hash>', 'admin');
