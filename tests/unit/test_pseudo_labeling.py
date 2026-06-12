"""
test_pseudo_labeling.py
=======================
Unit tests for pseudo-label generation pipeline.
Tests all three signals + fusion logic + mismatch label creation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pseudo_labeling.signal_rule_based import RuleBasedSeverityScorer, _extract_features
from src.pseudo_labeling.signal_resolution_time import ResolutionTimeSeverityScorer
from src.pseudo_labeling.fusion import fuse_signals, create_mismatch_label, PseudoLabelGenerator
from src.utils.config import COL_PRIORITY, COL_RESOLUTION_TIME


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame for testing."""
    return pd.DataFrame([
        {
            "Ticket ID": "TKT-001",
            "Ticket Subject": "Production server down",
            "Ticket Description": "Our production server is completely down. All users affected.",
            "Ticket Priority": "Low",
            "Ticket Channel": "Email",
            "Resolution Time": 2.0,
            "Ticket Type": "Technical Issue",
            "Customer Email": "a@b.com",
            "Product Purchased": "ProductA",
            "combined_text": "production server down our production server is completely down. all users affected.",
        },
        {
            "Ticket ID": "TKT-002",
            "Ticket Subject": "Change profile picture",
            "Ticket Description": "I want to update my profile photo.",
            "Ticket Priority": "Critical",
            "Ticket Channel": "Chat",
            "Resolution Time": 120.0,
            "Ticket Type": "Account Management",
            "Customer Email": "b@c.com",
            "Product Purchased": "ProductB",
            "combined_text": "change profile picture i want to update my profile photo.",
        },
        {
            "Ticket ID": "TKT-003",
            "Ticket Subject": "Login issue",
            "Ticket Description": "I cannot log in intermittently.",
            "Ticket Priority": "Medium",
            "Ticket Channel": "Phone",
            "Resolution Time": 8.0,
            "Ticket Type": "Bug Report",
            "Customer Email": "c@d.com",
            "Product Purchased": "ProductA",
            "combined_text": "login issue i cannot log in intermittently.",
        },
    ])


# ─────────────────────────────────────────────
# SIGNAL C: RULE-BASED TESTS
# ─────────────────────────────────────────────

class TestRuleBasedScorer:

    def test_critical_ticket_scores_high(self):
        text = "production server is completely down payment gateway failed all users affected"
        features = _extract_features(text)
        assert features["critical_kw_hits"] > 0
        assert features["business_impact_hits"] >= 0

    def test_low_ticket_scores_low(self):
        text = "i want to change my profile picture update notification preference"
        features = _extract_features(text)
        assert features["low_kw_hits"] > 0
        assert features["critical_kw_hits"] == 0

    def test_negation_reduces_score(self):
        text_positive = "server down payment failed critical outage"
        text_negated = "server is not down payment has not failed no critical issue"
        feats_pos = _extract_features(text_positive)
        feats_neg = _extract_features(text_negated)
        # Negated version should have lower or equal critical hits
        assert feats_neg["critical_kw_hits"] <= feats_pos["critical_kw_hits"]

    def test_scorer_adds_columns(self, sample_df):
        scorer = RuleBasedSeverityScorer()
        result = scorer.score(sample_df)
        assert "rule_severity" in result.columns
        assert "rule_score" in result.columns
        assert "rule_features" in result.columns

    def test_severity_values_valid(self, sample_df):
        scorer = RuleBasedSeverityScorer()
        result = scorer.score(sample_df)
        valid_levels = {"Low", "Medium", "High", "Critical"}
        assert set(result["rule_severity"].unique()).issubset(valid_levels)

    def test_scores_in_range(self, sample_df):
        scorer = RuleBasedSeverityScorer()
        result = scorer.score(sample_df)
        assert result["rule_score"].between(0.0, 1.0).all()

    def test_production_outage_classified_critical_or_high(self):
        scorer = RuleBasedSeverityScorer()
        result = scorer.score_single(
            "production server down all users affected payment gateway failed"
        )
        assert result["rule_severity"] in ["Critical", "High"]

    def test_profile_picture_classified_low(self):
        scorer = RuleBasedSeverityScorer()
        result = scorer.score_single("i want to change my profile picture")
        assert result["rule_severity"] in ["Low", "Medium"]


# ─────────────────────────────────────────────
# SIGNAL B: RESOLUTION TIME TESTS
# ─────────────────────────────────────────────

class TestResolutionTimeScorer:

    def test_fit_sets_thresholds(self, sample_df):
        scorer = ResolutionTimeSeverityScorer()
        scorer.fit(sample_df)
        assert scorer._fitted
        assert scorer._q_critical is not None
        assert scorer._q_high is not None
        assert scorer._q_medium is not None

    def test_score_adds_columns(self, sample_df):
        scorer = ResolutionTimeSeverityScorer()
        scorer.fit(sample_df)
        result = scorer.score(sample_df)
        assert "rt_severity" in result.columns
        assert "rt_score" in result.columns

    def test_fast_rt_maps_to_high_severity(self, sample_df):
        scorer = ResolutionTimeSeverityScorer()
        scorer.fit(sample_df)
        result = scorer.score(sample_df)
        # TKT-001 has RT=2h → should be Critical or High
        row = result[result["Ticket ID"] == "TKT-001"].iloc[0]
        assert row["rt_severity"] in ["Critical", "High"]

    def test_slow_rt_maps_to_low_severity(self, sample_df):
        scorer = ResolutionTimeSeverityScorer()
        scorer.fit(sample_df)
        result = scorer.score(sample_df)
        # TKT-002 has RT=120h → should be Low
        row = result[result["Ticket ID"] == "TKT-002"].iloc[0]
        assert row["rt_severity"] in ["Low", "Medium"]

    def test_scores_in_range(self, sample_df):
        scorer = ResolutionTimeSeverityScorer()
        scorer.fit(sample_df)
        result = scorer.score(sample_df)
        assert result["rt_score"].between(0.0, 1.0).all()

    def test_auto_fits_if_not_fitted(self, sample_df):
        scorer = ResolutionTimeSeverityScorer()
        # Should auto-fit without raising
        result = scorer.score(sample_df)
        assert "rt_severity" in result.columns


# ─────────────────────────────────────────────
# FUSION TESTS
# ─────────────────────────────────────────────

class TestFusionEngine:

    def test_unanimous_agreement_returns_that_severity(self):
        severity, confidence, agreement = fuse_signals(
            "Critical", 0.9, "Critical", 0.85, "Critical", 0.88
        )
        assert severity == "Critical"
        assert agreement == 1.0
        assert confidence > 0.5

    def test_majority_vote_wins(self):
        # 2 say Critical, 1 says Low
        severity, confidence, agreement = fuse_signals(
            "Critical", 0.9, "Critical", 0.85, "Low", 0.9
        )
        assert severity == "Critical"
        assert agreement == pytest.approx(2 / 3, abs=0.01)

    def test_weighted_preference_for_semantic(self):
        # Semantic has highest weight (0.45)
        # If semantic says Critical with high confidence, it should win
        severity, confidence, agreement = fuse_signals(
            "Critical", 0.95, "Low", 0.60, "Low", 0.60
        )
        # Semantic (0.45 * 0.95 = 0.4275) vs Low signals combined (0.25*0.60 + 0.30*0.60 = 0.33)
        assert severity == "Critical"

    def test_confidence_in_valid_range(self):
        for _ in range(10):
            sev, conf, agr = fuse_signals(
                np.random.choice(["Low", "Medium", "High", "Critical"]), np.random.random(),
                np.random.choice(["Low", "Medium", "High", "Critical"]), np.random.random(),
                np.random.choice(["Low", "Medium", "High", "Critical"]), np.random.random(),
            )
            assert 0.0 <= conf <= 1.0
            assert 0.0 <= agr <= 1.0


# ─────────────────────────────────────────────
# MISMATCH LABEL TESTS
# ─────────────────────────────────────────────

class TestMismatchLabel:

    def test_consistent_when_equal(self):
        label, mtype, delta = create_mismatch_label("High", "High")
        assert label == "Consistent"
        assert mtype == ""
        assert delta == 0

    def test_hidden_crisis_when_underpriorized(self):
        label, mtype, delta = create_mismatch_label("Low", "Critical")
        assert label == "Mismatch"
        assert mtype == "Hidden Crisis"
        assert delta > 0

    def test_false_alarm_when_overpriorized(self):
        label, mtype, delta = create_mismatch_label("Critical", "Low")
        assert label == "Mismatch"
        assert mtype == "False Alarm"
        assert delta < 0

    def test_delta_magnitude_correct(self):
        _, _, delta = create_mismatch_label("Low", "Critical")
        assert delta == 3  # Critical(3) - Low(0) = 3

    def test_all_severity_combinations(self):
        levels = ["Low", "Medium", "High", "Critical"]
        for assigned in levels:
            for inferred in levels:
                label, mtype, delta = create_mismatch_label(assigned, inferred)
                assert label in ["Consistent", "Mismatch"]
                if label == "Mismatch":
                    assert mtype in ["Hidden Crisis", "False Alarm"]
                    assert delta != 0


# ─────────────────────────────────────────────
# PSEUDO-LABEL GENERATOR INTEGRATION TEST
# ─────────────────────────────────────────────

class TestPseudoLabelGenerator:

    def test_generate_adds_all_required_columns(self, sample_df):
        plg = PseudoLabelGenerator()
        result = plg.generate(sample_df)

        required_cols = [
            "semantic_severity", "semantic_score",
            "rt_severity", "rt_score",
            "rule_severity", "rule_score",
            "inferred_severity", "fusion_confidence", "signal_agreement",
            "mismatch_label", "mismatch_type", "severity_delta", "label",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_label_is_binary(self, sample_df):
        plg = PseudoLabelGenerator()
        result = plg.generate(sample_df)
        assert set(result["label"].unique()).issubset({0, 1})

    def test_ablation_stats_has_pairs(self, sample_df):
        plg = PseudoLabelGenerator()
        result = plg.generate(sample_df)
        ablation = plg.get_ablation_stats(result)
        assert len(ablation) >= 3  # at least 3 pairs

    def test_known_hidden_crisis_detected(self, sample_df):
        """TKT-001: Low priority, production outage → should be Hidden Crisis."""
        plg = PseudoLabelGenerator()
        result = plg.generate(sample_df)
        row = result[result["Ticket ID"] == "TKT-001"].iloc[0]
        # At minimum, inferred severity should be higher than Low
        from src.utils.config import PRIORITY_MAP
        assert PRIORITY_MAP[row["inferred_severity"]] >= PRIORITY_MAP["Medium"]
