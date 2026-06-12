"""
fusion.py
=========
Signal Fusion Engine for Pseudo-Label Generation.

Fuses 3 independent severity signals into a final inferred severity:
  - Signal A: Semantic (sentence-transformers embeddings)
  - Signal B: Resolution Time (distributional analysis)
  - Signal C: Rule-Based NLP (keyword/pattern features)

Fusion Strategy:
  Weighted Voting with Confidence Gating:
  1. Each signal votes for a severity level with a confidence weight.
  2. Votes are aggregated using configured signal weights (A:0.45, B:0.25, C:0.30).
  3. Final severity = highest weighted vote.
  4. Pseudo-label confidence = agreement ratio across signals.

Mismatch Label Creation:
  Compares inferred_severity vs assigned_priority.
  Output: "Consistent" or "Mismatch" with subtype.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from src.utils.config import (
    SIGNAL_WEIGHTS, PRIORITY_LEVELS, PRIORITY_MAP, COL_PRIORITY,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_ORDER = PRIORITY_LEVELS  # ["Low", "Medium", "High", "Critical"]
SEVERITY_RANK = PRIORITY_MAP       # {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def fuse_signals(
    semantic_severity: str,
    semantic_score: float,
    rt_severity: str,
    rt_score: float,
    rule_severity: str,
    rule_score: float,
    weights: Dict[str, float] = None,
) -> Tuple[str, float, float]:
    """
    Fuses three severity signals into a single inferred severity.

    Mechanism:
      For each severity level, accumulate weighted confidence from
      each signal that voted for it. The level with highest total
      weighted confidence wins.

    Args:
        semantic_severity: Severity label from Signal A.
        semantic_score: Confidence from Signal A.
        rt_severity: Severity label from Signal B.
        rt_score: Confidence from Signal B.
        rule_severity: Severity label from Signal C.
        rule_score: Confidence from Signal C.
        weights: Optional override for signal weights.

    Returns:
        Tuple of (fused_severity, fused_confidence, signal_agreement).
          - fused_severity: Final inferred severity label.
          - fused_confidence: Weighted confidence [0, 1].
          - signal_agreement: Fraction of signals that agree [0, 1].
    """
    w = weights or SIGNAL_WEIGHTS
    votes: Dict[str, float] = {level: 0.0 for level in SEVERITY_ORDER}

    # Accumulate weighted confidence votes
    votes[semantic_severity] += w["semantic"] * semantic_score
    votes[rt_severity] += w["resolution_time"] * rt_score
    votes[rule_severity] += w["rule_based"] * rule_score

    # Winner
    total_weight = sum(votes.values()) + 1e-9
    fused_severity = max(votes, key=votes.get)
    fused_confidence = round(votes[fused_severity] / total_weight, 4)

    # Signal agreement: how many signals agree with the winner
    signals_voting = [semantic_severity, rt_severity, rule_severity]
    agreement_count = sum(1 for s in signals_voting if s == fused_severity)
    signal_agreement = round(agreement_count / len(signals_voting), 4)

    return fused_severity, fused_confidence, signal_agreement


def create_mismatch_label(
    assigned_priority: str,
    inferred_severity: str,
) -> Tuple[str, str, int]:
    """
    Creates binary mismatch label by comparing assigned vs inferred severity.

    Classification:
      - "Consistent": Assigned priority matches inferred severity.
      - "Mismatch":   Assigned priority conflicts with inferred severity.
        - "Hidden Crisis": inferred > assigned (under-prioritized)
        - "False Alarm":   inferred < assigned (over-prioritized)

    Args:
        assigned_priority: Human-assigned priority label.
        inferred_severity: System-inferred severity label.

    Returns:
        Tuple of (mismatch_label, mismatch_type, severity_delta).
          - mismatch_label: "Consistent" | "Mismatch"
          - mismatch_type:  "" | "Hidden Crisis" | "False Alarm"
          - severity_delta: Signed integer (inferred_rank - assigned_rank)
    """
    assigned_rank = SEVERITY_RANK.get(assigned_priority, 1)
    inferred_rank = SEVERITY_RANK.get(inferred_severity, 1)
    delta = inferred_rank - assigned_rank

    if delta == 0:
        return "Consistent", "", 0
    elif delta > 0:
        return "Mismatch", "Hidden Crisis", delta
    else:
        return "Mismatch", "False Alarm", delta


class PseudoLabelGenerator:
    """
    Orchestrates all three signals and generates pseudo-labels.

    Pipeline:
      1. Run Semantic Scorer (Signal A)
      2. Run Resolution Time Scorer (Signal B)
      3. Run Rule-Based Scorer (Signal C)
      4. Fuse signals
      5. Generate mismatch labels
      6. Compute signal agreement stats

    Usage:
        plg = PseudoLabelGenerator()
        labeled_df = plg.generate(df)
        stats = plg.get_ablation_stats(labeled_df)
    """

    def __init__(self, signal_weights: Dict[str, float] = None):
        from src.pseudo_labeling.signal_semantic import SemanticSeverityScorer
        from src.pseudo_labeling.signal_resolution_time import ResolutionTimeSeverityScorer
        from src.pseudo_labeling.signal_rule_based import RuleBasedSeverityScorer

        self.semantic_scorer = SemanticSeverityScorer()
        self.rt_scorer = ResolutionTimeSeverityScorer()
        self.rule_scorer = RuleBasedSeverityScorer()
        self.weights = signal_weights or SIGNAL_WEIGHTS

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pseudo-label generation pipeline.

        Args:
            df: Cleaned DataFrame from data_loader.

        Returns:
            DataFrame with all signal columns and final pseudo-labels:
              - semantic_severity, semantic_score
              - rt_severity, rt_score
              - rule_severity, rule_score
              - inferred_severity, fusion_confidence, signal_agreement
              - mismatch_label, mismatch_type, severity_delta
        """
        logger.info("=" * 60)
        logger.info("PSEUDO-LABEL GENERATION PIPELINE")
        logger.info("=" * 60)

        # Signal A: Semantic
        logger.info("Step 1/5: Running Semantic Severity Scorer...")
        df = self.semantic_scorer.score(df)

        # Signal B: Resolution Time
        logger.info("Step 2/5: Running Resolution Time Scorer...")
        self.rt_scorer.fit(df)
        df = self.rt_scorer.score(df)

        # Signal C: Rule-Based
        logger.info("Step 3/5: Running Rule-Based Scorer...")
        df = self.rule_scorer.score(df)

        # Signal Fusion
        logger.info("Step 4/5: Fusing signals...")
        fused_results = [
            fuse_signals(
                row["semantic_severity"], row["semantic_score"],
                row["rt_severity"], row["rt_score"],
                row["rule_severity"], row["rule_score"],
                self.weights,
            )
            for _, row in df.iterrows()
        ]
        df["inferred_severity"] = [r[0] for r in fused_results]
        df["fusion_confidence"] = [r[1] for r in fused_results]
        df["signal_agreement"] = [r[2] for r in fused_results]

        # Mismatch Labels
        logger.info("Step 5/5: Creating mismatch labels...")
        mismatch_results = [
            create_mismatch_label(row[COL_PRIORITY], row["inferred_severity"])
            for _, row in df.iterrows()
        ]
        df["mismatch_label"] = [r[0] for r in mismatch_results]
        df["mismatch_type"] = [r[1] for r in mismatch_results]
        df["severity_delta"] = [r[2] for r in mismatch_results]

        # Binary label for classifier
        df["label"] = (df["mismatch_label"] == "Mismatch").astype(int)

        logger.info("=" * 60)
        logger.info("PSEUDO-LABEL GENERATION COMPLETE")
        logger.info(f"Total tickets:  {len(df)}")
        logger.info(f"Mismatches:     {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
        logger.info(f"Hidden Crisis:  {(df['mismatch_type']=='Hidden Crisis').sum()}")
        logger.info(f"False Alarm:    {(df['mismatch_type']=='False Alarm').sum()}")
        logger.info("=" * 60)
        return df

    def get_ablation_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes pairwise signal agreement for ablation study.

        For each pair of signals, computes:
          - Pairwise agreement rate
          - Agreement with fused label

        Args:
            df: DataFrame after generate() has been called.

        Returns:
            Ablation stats DataFrame.
        """
        pairs = [
            ("semantic_severity", "rt_severity", "Semantic vs RT"),
            ("semantic_severity", "rule_severity", "Semantic vs Rule"),
            ("rt_severity", "rule_severity", "RT vs Rule"),
            ("semantic_severity", "inferred_severity", "Semantic vs Fused"),
            ("rt_severity", "inferred_severity", "RT vs Fused"),
            ("rule_severity", "inferred_severity", "Rule vs Fused"),
        ]
        rows = []
        for col_a, col_b, label in pairs:
            agreement = (df[col_a] == df[col_b]).mean()
            rows.append({"Signal Pair": label, "Agreement Rate": round(agreement, 4)})
        ablation_df = pd.DataFrame(rows)
        logger.info(f"\nAblation Study:\n{ablation_df.to_string(index=False)}")
        return ablation_df

    def get_signal_contribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shows how often each signal is the 'deciding' factor
        (i.e., agrees with final fused label when others don't).

        Args:
            df: DataFrame after generate() has been called.

        Returns:
            Signal contribution DataFrame.
        """
        contributions = {}
        for signal_col in ["semantic_severity", "rt_severity", "rule_severity"]:
            # Count cases where THIS signal agrees with fused AND other two don't
            other_signals = [s for s in ["semantic_severity", "rt_severity", "rule_severity"]
                             if s != signal_col]
            decisive = (
                (df[signal_col] == df["inferred_severity"]) &
                (df[other_signals[0]] != df["inferred_severity"]) &
                (df[other_signals[1]] != df["inferred_severity"])
            ).sum()
            contributions[signal_col.replace("_severity", "")] = decisive

        contrib_df = pd.DataFrame(
            list(contributions.items()), columns=["Signal", "Decisive Votes"]
        )
        contrib_df["Contribution %"] = (
            contrib_df["Decisive Votes"] / contrib_df["Decisive Votes"].sum() * 100
        ).round(1)
        return contrib_df
