"""
data_loader.py
==============
Handles dataset loading, validation, and preprocessing for SIA.
Generates a synthetic dataset when the real Kaggle CSV is absent
so the pipeline can run end-to-end without manual download.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.config import (
    DATASET_PATH, COL_TICKET_ID, COL_SUBJECT, COL_DESCRIPTION,
    COL_PRIORITY, COL_CHANNEL, COL_RESOLUTION_TIME, COL_TICKET_TYPE,
    COL_CUSTOMER_EMAIL, COL_PRODUCT, PRIORITY_LEVELS, RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────

SYNTHETIC_TEMPLATES = [
    # (subject, description, priority, resolution_hours)
    ("Production payment gateway down", "Our production payment gateway is completely down and customers cannot complete purchases. Revenue impact is severe.", "Low", 1.5),
    ("Server outage affecting all users", "The main application server has crashed. All users are locked out. Immediate action required.", "Medium", 2.0),
    ("Security breach detected", "We have detected unauthorized access to customer credentials. Data breach in progress.", "Low", 3.0),
    ("Database corruption", "Primary database is corrupted. All data access is failing and backups are 24 hours old.", "Medium", 4.0),
    ("Change profile picture", "I want to update my profile photo on the account page.", "Critical", 120.0),
    ("Update notification preferences", "Can you help me turn off email notifications? I get too many.", "High", 96.0),
    ("Feature request: dark mode", "It would be nice to have a dark mode option in the UI.", "Critical", 150.0),
    ("Password reset email not received", "I requested a password reset 10 minutes ago but haven't received the email yet.", "Medium", 8.0),
    ("API returning 500 errors intermittently", "Our integration is getting 500 errors on about 30% of requests since this morning.", "Low", 6.0),
    ("Login page not loading", "Cannot access the login page at all. Getting a blank screen. Entire team is blocked.", "Low", 3.5),
    ("Minor UI alignment issue", "The submit button is slightly off-center on mobile devices.", "High", 200.0),
    ("Billing inquiry", "I have a question about my invoice from last month.", "Critical", 72.0),
    ("Data loss after update", "After the latest update, all my saved configurations are gone. This is a production system.", "Low", 5.0),
    ("Slow dashboard loading", "The analytics dashboard takes 15 seconds to load. Used to be instant.", "Medium", 24.0),
    ("Email change request", "I'd like to change my registered email address.", "High", 48.0),
    ("Complete service outage", "All services are down for our enterprise account. Entire company is affected.", "Medium", 1.0),
    ("SLA breach notification", "We are breaching our SLA with key clients due to ongoing outage. Legal escalation imminent.", "Low", 2.0),
    ("Font size preference", "Could the default font size be increased slightly?", "Critical", 168.0),
    ("OAuth integration broken", "OAuth login is returning 401 for all users. Nobody can sign in via SSO.", "Medium", 4.5),
    ("Ransomware detected on server", "Ransomware has encrypted our backup server. Customer data at risk.", "Low", 1.0),
    ("How do I export data?", "I'm not sure how to export my data to CSV format.", "Medium", 36.0),
    ("Transaction failures 100%", "100% of payment transactions are failing. No orders can be processed.", "Low", 1.5),
    ("Cosmetic bug on profile page", "There's a small spacing issue between the avatar and username.", "High", 180.0),
    ("Cannot access admin panel", "The admin dashboard is returning 403 for our entire admin team since the deployment.", "Low", 3.0),
    ("Feedback on new design", "The new interface looks great! Just wanted to share some minor suggestions.", "Critical", 200.0),
    ("Production API rate limit hit", "Our production API is being rate-limited causing customer-facing errors.", "Low", 2.0),
    ("Request for documentation", "Could you point me to the API documentation?", "High", 72.0),
    ("Critical memory leak", "Application memory usage grows until crash every 2 hours in production.", "Low", 4.0),
    ("Update company logo", "We rebranded and need to update the logo in the portal.", "Critical", 120.0),
    ("Compliance audit failure", "System failed a SOC2 audit due to logging gaps. Regulatory action pending.", "Low", 6.0),
    ("Typo on website", "There's a typo on the About Us page.", "Critical", 96.0),
    ("Users cannot checkout", "The checkout button is broken. Customers cannot complete purchases. Losing thousands per minute.", "Low", 1.0),
    ("Wrong currency displayed", "Prices show in USD but should show in EUR for our European customers.", "Medium", 12.0),
    ("Forgot username", "I forgot what email I used to sign up.", "High", 48.0),
    ("Zero downtime migration failed", "The database migration script failed mid-way. Production DB is in inconsistent state.", "Low", 2.5),
    ("Suggestion for better UX", "The onboarding flow could be simplified for new users.", "High", 100.0),
    ("SSL certificate expired", "SSL certificate expired. All HTTPS requests are failing. Browser warnings everywhere.", "Medium", 1.0),
    ("Webhook delivery failing", "Webhooks are not being delivered to our endpoints. Critical business automation broken.", "Low", 5.0),
    ("Change account timezone", "I'd like to change my account timezone setting.", "Critical", 48.0),
    ("Intrusion detection alert", "Multiple failed login attempts from foreign IPs. Possible brute force attack in progress.", "Low", 2.0),
]

CHANNELS = ["Email", "Chat", "Phone", "Social Media", "Portal"]
TICKET_TYPES = ["Technical Issue", "Billing", "Feature Request", "Account Management", "Bug Report"]
PRODUCTS = ["ProductA", "ProductB", "SaaS Platform", "Enterprise Suite", "Mobile App"]


def generate_synthetic_dataset(n_tickets: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a synthetic support ticket dataset when Kaggle CSV is unavailable.
    Based on realistic patterns with intentional mismatches injected.

    Args:
        n_tickets: Number of tickets to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ticket data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tickets):
        tmpl = SYNTHETIC_TEMPLATES[i % len(SYNTHETIC_TEMPLATES)]
        subject, description, priority, res_hours = tmpl

        # Add slight noise to resolution time
        res_noise = rng.normal(1.0, 0.2)
        resolution_time = max(0.5, res_hours * res_noise)

        rows.append({
            COL_TICKET_ID: f"TKT-{i+1:05d}",
            COL_SUBJECT: subject,
            COL_DESCRIPTION: description,
            COL_PRIORITY: priority,
            COL_CHANNEL: rng.choice(CHANNELS),
            COL_RESOLUTION_TIME: round(resolution_time, 2),
            COL_TICKET_TYPE: rng.choice(TICKET_TYPES),
            COL_CUSTOMER_EMAIL: f"user{i+1}@example.com",
            COL_PRODUCT: rng.choice(PRODUCTS),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Generated synthetic dataset: {len(df)} tickets.")
    return df


# ─────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────

def load_dataset(path: Optional[Path] = None, synthetic_fallback: bool = True) -> pd.DataFrame:
    """
    Loads the customer support ticket dataset.
    Falls back to synthetic data if CSV is not found.

    Args:
        path: Optional override path to CSV.
        synthetic_fallback: If True, generate synthetic data when file not found.

    Returns:
        Validated DataFrame.
    """
    target = path or DATASET_PATH
    if target.exists():
        logger.info(f"Loading dataset from: {target}")
        df = pd.read_csv(target)
    elif synthetic_fallback:
        logger.warning(f"Dataset not found at {target}. Generating synthetic dataset.")
        df = generate_synthetic_dataset()
    else:
        raise FileNotFoundError(f"Dataset not found at {target}")

    df = validate_and_clean(df)
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans the raw dataset.

    Steps:
    - Ensures required columns are present (or creates them).
    - Standardizes priority labels.
    - Fills missing text with empty string.
    - Creates combined text field.
    - Resets index.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Validating and cleaning dataset...")
    original_len = len(df)

    # Ensure ticket ID column
    if COL_TICKET_ID not in df.columns:
        df[COL_TICKET_ID] = [f"TKT-{i+1:05d}" for i in range(len(df))]

    # Fill missing text
    for col in [COL_SUBJECT, COL_DESCRIPTION]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""

    # Standardize priority
    if COL_PRIORITY in df.columns:
        df[COL_PRIORITY] = df[COL_PRIORITY].astype(str).str.strip().str.capitalize()
        df[COL_PRIORITY] = df[COL_PRIORITY].replace({
            "Urgent": "Critical", "Very high": "Critical",
            "Normal": "Medium", "Low priority": "Low",
        })
        df = df[df[COL_PRIORITY].isin(PRIORITY_LEVELS)].copy()
    else:
        df[COL_PRIORITY] = "Medium"

    # Resolution time — numeric
    if COL_RESOLUTION_TIME in df.columns:
        df[COL_RESOLUTION_TIME] = pd.to_numeric(df[COL_RESOLUTION_TIME], errors="coerce")
        median_rt = df[COL_RESOLUTION_TIME].median()
        df[COL_RESOLUTION_TIME] = df[COL_RESOLUTION_TIME].fillna(median_rt)
    else:
        df[COL_RESOLUTION_TIME] = 24.0  # default 24 hours

    # Optional columns
    for col in [COL_CHANNEL, COL_TICKET_TYPE, COL_CUSTOMER_EMAIL, COL_PRODUCT]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Combined text for NLP
    df["combined_text"] = (
        df[COL_SUBJECT].str.lower() + " " + df[COL_DESCRIPTION].str.lower()
    ).str.strip()

    df = df.reset_index(drop=True)
    logger.info(f"Cleaned dataset: {original_len} → {len(df)} rows.")
    return df
