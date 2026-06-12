"""
signal_rule_based.py
====================
Signal C: Rule-Based Urgency Scoring via NLP Feature Extraction.

Features:
  - Critical / High / Medium / Low keyword density scoring
  - Escalation phrase detection
  - Negation-aware keyword matching
  - Business-impact indicator detection
  - Exclamation / capitalization urgency markers
  - Sentence length and density signals

Output columns:
  - rule_severity:    "Low" | "Medium" | "High" | "Critical"
  - rule_score:       float in [0, 1]
  - rule_features:    dict with individual signal breakdown
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from src.utils.config import (
    CRITICAL_KEYWORDS, HIGH_KEYWORDS, MEDIUM_KEYWORDS, LOW_KEYWORDS,
    ESCALATION_PHRASES, NEGATION_WORDS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# ADDITIONAL RULE PATTERNS
# ─────────────────────────────────────────────

BUSINESS_IMPACT_PATTERNS = [
    r"revenue (loss|impact|drop)",
    r"customers? (cannot|can't|unable to)",
    r"all users? (affected|blocked|locked)",
    r"entire (company|team|organization)",
    r"(legal|regulatory|compliance) (action|violation|issue)",
    r"sla (breach|violation|penalty)",
    r"(losing|lost) (money|revenue|customers?)",
    r"financial impact",
    r"business critical",
]

QUANTIFIER_PATTERNS = [
    r"\b100%\b",
    r"\ball\b.{0,20}\b(down|fail|broken|crash)",
    r"\bno (one|users?|customers?)\b.{0,20}(access|login|purchase)",
    r"\beveryone\b.{0,20}(affected|blocked)",
]

NEGATION_NEUTRALIZER = re.compile(
    r"\b(" + "|".join(NEGATION_WORDS) + r")\b\s+\w+\s+", re.IGNORECASE
)


def _preprocess_text(text: str) -> str:
    """Lowercase, strip HTML, normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text.lower().strip())
    return text


def _count_keyword_hits(text: str, keywords: List[str], negation_aware: bool = True) -> int:
    """
    Counts keyword hits in text with optional negation detection.

    Negation window: if a negation word appears within 3 tokens before
    the keyword, the hit is discounted.

    Args:
        text: Preprocessed lowercase text.
        keywords: List of keyword strings.
        negation_aware: Whether to apply negation discounting.

    Returns:
        Weighted hit count (negated hits count as 0.2).
    """
    score = 0.0
    for kw in keywords:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b")
        for match in pattern.finditer(text):
            start = match.start()
            # Check for negation in a 40-char window before the keyword
            context_window = text[max(0, start - 40): start]
            negated = any(
                re.search(r"\b" + neg + r"\b", context_window, re.IGNORECASE)
                for neg in NEGATION_WORDS
            )
            if negated and negation_aware:
                score += 0.2  # Discounted — negated urgency
            else:
                score += 1.0
    return score


def _extract_features(text: str) -> Dict[str, float]:
    """
    Extracts all rule-based urgency features from ticket text.

    Args:
        text: Raw combined text (subject + description).

    Returns:
        Dict with feature names and scores.
    """
    clean = _preprocess_text(text)
    raw = text  # keep original for caps detection

    features = {}

    # ── Keyword hits ──────────────────────────────────────
    features["critical_kw_hits"] = _count_keyword_hits(clean, CRITICAL_KEYWORDS)
    features["high_kw_hits"] = _count_keyword_hits(clean, HIGH_KEYWORDS)
    features["medium_kw_hits"] = _count_keyword_hits(clean, MEDIUM_KEYWORDS)
    features["low_kw_hits"] = _count_keyword_hits(clean, LOW_KEYWORDS)

    # ── Escalation phrases ────────────────────────────────
    features["escalation_hits"] = sum(
        1 for phrase in ESCALATION_PHRASES
        if re.search(r"\b" + re.escape(phrase) + r"\b", clean)
    )

    # ── Business impact patterns ──────────────────────────
    features["business_impact_hits"] = sum(
        1 for pattern in BUSINESS_IMPACT_PATTERNS
        if re.search(pattern, clean)
    )

    # ── Quantifier patterns ───────────────────────────────
    features["quantifier_hits"] = sum(
        1 for pattern in QUANTIFIER_PATTERNS
        if re.search(pattern, clean)
    )

    # ── ALL CAPS words (urgency marker) ───────────────────
    caps_words = re.findall(r"\b[A-Z]{3,}\b", raw)
    features["caps_word_count"] = len(caps_words)

    # ── Exclamation marks ─────────────────────────────────
    features["exclamation_count"] = raw.count("!")

    # ── Text length signal (longer = more detailed = possibly more severe) ──
    features["text_length_norm"] = min(len(clean) / 500.0, 1.0)

    return features


def _features_to_severity(features: Dict[str, float]) -> Tuple[str, float]:
    """
    Converts extracted features into a severity label and confidence score
    using a weighted scoring system.

    Scoring logic:
      Critical score = critical_kw * 3 + business_impact * 2 + quantifier * 2 + escalation * 1.5
      High score     = high_kw * 2 + escalation * 1 + caps * 0.5
      Medium score   = medium_kw * 1.5
      Low score      = low_kw * 2 + (no high/critical hits)

    Confidence = normalized margin between top score and second score.

    Args:
        features: Dict from _extract_features.

    Returns:
        Tuple (severity_label, confidence_score).
    """
    scores = {
        "Critical": (
            features["critical_kw_hits"] * 3.0
            + features["business_impact_hits"] * 2.0
            + features["quantifier_hits"] * 2.0
            + features["escalation_hits"] * 1.5
            + features["caps_word_count"] * 0.3
            + features["exclamation_count"] * 0.2
        ),
        "High": (
            features["high_kw_hits"] * 2.0
            + features["escalation_hits"] * 1.0
            + features["caps_word_count"] * 0.4
            + features["exclamation_count"] * 0.3
        ),
        "Medium": (
            features["medium_kw_hits"] * 1.5
            + features["text_length_norm"] * 0.5
        ),
        "Low": (
            features["low_kw_hits"] * 2.0
            + (1.5 if features["critical_kw_hits"] == 0 and features["high_kw_hits"] == 0 else 0)
        ),
    }

    total = sum(scores.values()) + 1e-9
    normalized = {k: v / total for k, v in scores.items()}

    best = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    # Confidence: based on margin and normalized score
    confidence = min(0.95, 0.50 + (margin / (total + 1e-9)) * 2.0)
    confidence = max(0.40, confidence)

    return best, round(confidence, 4)


class RuleBasedSeverityScorer:
    """
    Rule-based severity scorer using NLP feature extraction.

    Fully interpretable — every decision is traceable to specific features.
    No model loading required — runs instantly.

    Usage:
        scorer = RuleBasedSeverityScorer()
        df = scorer.score(df)
        features_df = scorer.get_features_df(df)
    """

    def score_single(self, text: str) -> Dict:
        """
        Scores a single ticket text.

        Args:
            text: Combined ticket text.

        Returns:
            Dict with severity, score, and feature breakdown.
        """
        features = _extract_features(text)
        severity, confidence = _features_to_severity(features)
        return {
            "rule_severity": severity,
            "rule_score": confidence,
            "rule_features": features,
        }

    def score(self, df: pd.DataFrame, text_col: str = "combined_text") -> pd.DataFrame:
        """
        Adds rule-based severity columns to the DataFrame.

        Args:
            df: Input DataFrame.
            text_col: Column with combined ticket text.

        Returns:
            DataFrame with added columns:
              - rule_severity
              - rule_score
              - rule_features (dict serialized per row)
        """
        logger.info(f"Running rule-based scoring on {len(df)} tickets...")
        df = df.copy()

        results = [self.score_single(text) for text in df[text_col]]
        df["rule_severity"] = [r["rule_severity"] for r in results]
        df["rule_score"] = [r["rule_score"] for r in results]
        df["rule_features"] = [r["rule_features"] for r in results]

        logger.info(
            f"Rule-based scoring complete. Distribution:\n"
            f"{pd.Series(df['rule_severity']).value_counts().to_string()}"
        )
        return df

    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes average feature values per severity level for analysis.

        Args:
            df: DataFrame with rule_features and rule_severity columns.

        Returns:
            DataFrame of feature means by severity.
        """
        if "rule_features" not in df.columns:
            raise ValueError("Must call score() before get_feature_importance().")

        feature_df = pd.DataFrame(df["rule_features"].tolist())
        feature_df["rule_severity"] = df["rule_severity"].values
        return feature_df.groupby("rule_severity").mean().round(3)
