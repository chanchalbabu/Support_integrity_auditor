"""
config.py
=========
Central configuration for the Support Integrity Auditor (SIA).
All hyperparameters, paths, and constants live here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PSEUDO_LABEL_DIR = DATA_DIR / "pseudo_labels"
MODELS_DIR = BASE_DIR / "models"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
ADVANCED_MODEL_DIR = MODELS_DIR / "advanced"
OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
DOSSIERS_DIR = OUTPUTS_DIR / "dossiers"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# ─────────────────────────────────────────────
# DATASET CONFIG
# ─────────────────────────────────────────────
DATASET_FILENAME = "customer_support_tickets.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

# Column names as they appear in the dataset
COL_TICKET_ID = "Ticket ID"
COL_SUBJECT = "Ticket Subject"
COL_DESCRIPTION = "Ticket Description"
COL_PRIORITY = "Ticket Priority"
COL_CHANNEL = "Ticket Channel"
COL_RESOLUTION_TIME = "Resolution Time"
COL_TICKET_TYPE = "Ticket Type"
COL_CUSTOMER_EMAIL = "Customer Email"
COL_PRODUCT = "Product Purchased"

# Priority levels (ordered low → high)
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_MAP_INV = {v: k for k, v in PRIORITY_MAP.items()}

# ─────────────────────────────────────────────
# PSEUDO-LABEL GENERATION
# ─────────────────────────────────────────────

# Signal weights for fusion (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "semantic": 0.45,
    "resolution_time": 0.25,
    "rule_based": 0.30,
}

# Sentence transformer model
SBERT_MODEL = "all-MiniLM-L6-v2"

# Resolution time thresholds (in hours) → severity
RESOLUTION_TIME_THRESHOLDS = {
    "Critical": 4,    # resolved in < 4 hrs  → was critical
    "High": 12,       # resolved in < 12 hrs → was high
    "Medium": 48,     # resolved in < 48 hrs → was medium
    "Low": float("inf"),  # anything above
}

# Rule-based keyword lexicon
CRITICAL_KEYWORDS = [
    "server down", "production outage", "payment failed", "payment gateway",
    "security breach", "data loss", "data breach", "system failure",
    "complete outage", "cannot access", "database down", "critical failure",
    "service unavailable", "complete downtime", "emergency", "urgent",
    "immediately", "asap", "all users affected", "entire system",
    "revenue loss", "financial impact", "sla breach", "compliance violation",
    "ransomware", "hack", "intrusion", "credentials compromised",
]

HIGH_KEYWORDS = [
    "not working", "broken", "error", "bug", "crash", "slow", "performance",
    "degraded", "intermittent", "partial outage", "some users", "login issue",
    "authentication", "cannot login", "feature broken", "api error",
    "timeout", "connection refused", "502", "503", "500 error",
]

MEDIUM_KEYWORDS = [
    "help", "issue", "problem", "question", "inquiry", "request",
    "not sure", "confused", "clarification", "assistance", "support",
    "how to", "unable to", "unexpected behavior", "wrong result",
]

LOW_KEYWORDS = [
    "feature request", "suggestion", "feedback", "profile", "settings",
    "change password", "update email", "notification", "preference",
    "cosmetic", "ui improvement", "minor", "nice to have",
    "change picture", "profile picture", "username change",
]

ESCALATION_PHRASES = [
    "escalate", "manager", "legal action", "report", "lawsuit",
    "unacceptable", "this is ridiculous", "worst", "terrible",
    "not the first time", "repeatedly", "still not fixed",
]

NEGATION_WORDS = ["not", "never", "no", "cannot", "can't", "won't", "doesn't", "don't"]

# ─────────────────────────────────────────────
# ML PIPELINE CONFIG
# ─────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Baseline (TF-IDF + LR)
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 3)
LR_MAX_ITER = 1000
LR_C = 1.0

# Advanced (DeBERTa)
DEBERTA_MODEL_NAME = "microsoft/deberta-v3-small"
DEBERTA_MAX_LENGTH = 256
DEBERTA_BATCH_SIZE = 16
DEBERTA_LEARNING_RATE = 2e-5
DEBERTA_NUM_EPOCHS = 5
DEBERTA_WARMUP_RATIO = 0.1
DEBERTA_WEIGHT_DECAY = 0.01
DEBERTA_EARLY_STOPPING_PATIENCE = 2

# ─────────────────────────────────────────────
# EVALUATION THRESHOLDS
# ─────────────────────────────────────────────
MIN_ACCURACY = 0.83
MIN_MACRO_F1 = 0.82
MIN_PER_CLASS_RECALL = 0.78

# ─────────────────────────────────────────────
# DOSSIER CONFIG
# ─────────────────────────────────────────────
DOSSIER_VERSION = "1.0"
MAX_EVIDENCE_ITEMS = 5
MIN_CONFIDENCE_THRESHOLD = 0.60  # below this → "Low Confidence"
