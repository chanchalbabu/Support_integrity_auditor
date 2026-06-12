"""
signal_resolution_time.py
=========================
Signal B: Resolution-Time Severity Scoring.

Rationale:
  Critical issues are resolved fastest (high urgency).
  Low-severity issues take the longest (no rush).
  Resolution time is therefore an INVERSE severity proxy.

Approach:
  1. Fit a log-normal distribution to observed resolution times.
  2. Map percentiles → severity bins using calibrated thresholds.
  3. Short resolution time → higher inferred severity.

Output columns added:
  - rt_severity:  "Low" | "Medium" | "High" | "Critical"
  - rt_score:     float in [0, 1]
  - rt_percentile: position in distribution (0 = slowest, 1 = fastest)
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils.config import COL_RESOLUTION_TIME, RESOLUTION_TIME_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResolutionTimeSeverityScorer:
    """
    Infers ticket severity from resolution time using distributional analysis.

    The key insight: in a well-functioning support system, Critical tickets
    are escalated and resolved quickly. Low-priority tickets can wait days.
    Therefore, unusually short resolution times indicate the actual issue
    was treated as critical by support staff (regardless of assigned priority).

    Usage:
        scorer = ResolutionTimeSeverityScorer()
        scorer.fit(df)
        df = scorer.score(df)
    """

    def __init__(self):
        self._q_critical = None  # top X% fastest
        self._q_high = None
        self._q_medium = None
        self._fitted = False

    def fit(self, df: pd.DataFrame, rt_col: str = COL_RESOLUTION_TIME) -> "ResolutionTimeSeverityScorer":
        """
        Fits resolution time distribution from training data.

        Uses log-transformed percentiles to handle skewed distribution.

        Args:
            df: DataFrame with resolution time column.
            rt_col: Name of the resolution time column.

        Returns:
            self (for chaining).
        """
        rt = df[rt_col].dropna().values
        rt = rt[rt > 0]  # exclude zero/negative

        # Log-transform to normalize the heavily right-skewed distribution
        log_rt = np.log1p(rt)

        # Define percentile boundaries
        # Bottom 15% fastest → Critical
        # 15–35% → High
        # 35–65% → Medium
        # Top 35% slowest → Low
        self._q_critical = np.percentile(log_rt, 15)
        self._q_high = np.percentile(log_rt, 35)
        self._q_medium = np.percentile(log_rt, 65)

        self._fitted = True
        logger.info(
            f"Resolution time distribution fitted.\n"
            f"  Critical threshold (≤ {np.expm1(self._q_critical):.1f}h)\n"
            f"  High threshold    (≤ {np.expm1(self._q_high):.1f}h)\n"
            f"  Medium threshold  (≤ {np.expm1(self._q_medium):.1f}h)"
        )
        return self

    def _map_to_severity(self, log_rt_val: float) -> Tuple[str, float]:
        """
        Maps a single log-resolution-time value to severity + confidence.

        Args:
            log_rt_val: Log-transformed resolution time.

        Returns:
            Tuple of (severity_label, confidence_score).
        """
        if log_rt_val <= self._q_critical:
            severity = "Critical"
            # Confidence: how far below critical threshold
            margin = (self._q_critical - log_rt_val) / (self._q_critical + 1e-9)
            confidence = min(0.95, 0.70 + margin * 0.25)
        elif log_rt_val <= self._q_high:
            severity = "High"
            margin = (self._q_high - log_rt_val) / (self._q_high - self._q_critical + 1e-9)
            confidence = min(0.90, 0.60 + margin * 0.30)
        elif log_rt_val <= self._q_medium:
            severity = "Medium"
            margin = (self._q_medium - log_rt_val) / (self._q_medium - self._q_high + 1e-9)
            confidence = min(0.85, 0.55 + margin * 0.25)
        else:
            severity = "Low"
            margin = (log_rt_val - self._q_medium) / (log_rt_val + 1e-9)
            confidence = min(0.85, 0.55 + margin * 0.20)

        return severity, round(float(confidence), 4)

    def score(self, df: pd.DataFrame, rt_col: str = COL_RESOLUTION_TIME) -> pd.DataFrame:
        """
        Adds resolution-time severity columns to the DataFrame.

        Args:
            df: Input DataFrame.
            rt_col: Resolution time column name.

        Returns:
            DataFrame with added columns:
              - rt_severity
              - rt_score
              - rt_log
        """
        if not self._fitted:
            logger.info("Scorer not fitted — fitting on current data.")
            self.fit(df, rt_col)

        df = df.copy()
        rt_values = df[rt_col].fillna(df[rt_col].median()).values
        rt_values = np.where(rt_values <= 0, 0.1, rt_values)
        log_rt = np.log1p(rt_values)

        severities = []
        confidences = []
        for lv in log_rt:
            sev, conf = self._map_to_severity(lv)
            severities.append(sev)
            confidences.append(conf)

        df["rt_severity"] = severities
        df["rt_score"] = confidences
        df["rt_log"] = log_rt

        logger.info(
            f"Resolution-time scoring complete. Distribution:\n"
            f"{pd.Series(severities).value_counts().to_string()}"
        )
        return df
