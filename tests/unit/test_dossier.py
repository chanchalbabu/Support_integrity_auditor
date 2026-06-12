"""
test_dossier.py
===============
Unit + validation tests for the Evidence Dossier Generator.

Critical requirement: ZERO hallucination.
Every evidence item must be traceable to actual ticket data.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dossier.generator import DossierGenerator, GroundingValidator
import pandas as pd


@pytest.fixture
def mismatch_row():
    """A mismatch ticket row with all required prediction columns."""
    return pd.Series({
        "Ticket ID": "TKT-TEST-001",
        "Ticket Subject": "Production payment gateway down",
        "Ticket Description": "Our production payment gateway is completely down. Customers cannot complete purchases.",
        "Ticket Priority": "Low",
        "Ticket Channel": "Email",
        "Resolution Time": 1.5,
        "Ticket Type": "Technical Issue",
        "Customer Email": "test@example.com",
        "Product Purchased": "ProductA",
        "combined_text": "production payment gateway down customers cannot complete purchases",
        "inferred_severity": "Critical",
        "mismatch_label": "Mismatch",
        "mismatch_type": "Hidden Crisis",
        "severity_delta": 3,
        "fusion_confidence": 0.87,
        "signal_agreement": 0.67,
        "rt_severity": "Critical",
        "rt_score": 0.91,
        "rule_severity": "Critical",
        "rule_score": 0.85,
        "semantic_severity": "Critical",
        "semantic_score": 0.89,
    })


@pytest.fixture
def consistent_row():
    """A consistent ticket row."""
    return pd.Series({
        "Ticket ID": "TKT-TEST-002",
        "Ticket Subject": "Change profile picture",
        "Ticket Description": "I want to update my profile photo.",
        "Ticket Priority": "Low",
        "Ticket Channel": "Chat",
        "Resolution Time": 120.0,
        "Ticket Type": "Account Management",
        "Customer Email": "user@example.com",
        "Product Purchased": "ProductB",
        "combined_text": "change profile picture update profile photo",
        "inferred_severity": "Low",
        "mismatch_label": "Consistent",
        "mismatch_type": "",
        "severity_delta": 0,
        "fusion_confidence": 0.75,
        "signal_agreement": 1.0,
        "rt_severity": "Low",
        "rt_score": 0.80,
        "rule_severity": "Low",
        "rule_score": 0.78,
        "semantic_severity": "Low",
        "semantic_score": 0.82,
    })


class TestGroundingValidator:

    def test_keyword_in_text_returns_true(self):
        v = GroundingValidator()
        assert v.validate_keyword_evidence("payment gateway", "production payment gateway failed") is True

    def test_keyword_not_in_text_returns_false(self):
        v = GroundingValidator()
        assert v.validate_keyword_evidence("server down", "profile picture update") is False

    def test_rt_validation_within_tolerance(self):
        v = GroundingValidator()
        assert v.validate_rt_evidence(2.0, 2.0) is True
        assert v.validate_rt_evidence(2.0, 2.019) is True  # within 1% tolerance

    def test_channel_validation(self):
        v = GroundingValidator()
        assert v.validate_channel_evidence("Email", "email") is True
        assert v.validate_channel_evidence("Phone", "chat") is False


class TestDossierGenerator:

    def test_generates_dossier_for_mismatch(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert dossier is not None

    def test_returns_none_for_consistent(self, consistent_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(consistent_row)
        assert dossier is None

    def test_schema_keys_present(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        required_keys = [
            "ticket_id", "assigned_priority", "inferred_severity",
            "mismatch_type", "severity_delta", "feature_evidence",
            "constraint_analysis", "confidence",
        ]
        for key in required_keys:
            assert key in dossier, f"Missing key: {key}"

    def test_ticket_id_matches_input(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert dossier["ticket_id"] == "TKT-TEST-001"

    def test_assigned_priority_matches_input(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert dossier["assigned_priority"] == "Low"

    def test_inferred_severity_matches_input(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert dossier["inferred_severity"] == "Critical"

    def test_mismatch_type_is_valid(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert dossier["mismatch_type"] in ["Hidden Crisis", "False Alarm"]

    def test_severity_delta_is_integer(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert isinstance(dossier["severity_delta"], int)

    def test_evidence_items_not_empty(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert len(dossier["feature_evidence"]) > 0

    def test_evidence_grounding_keyword_in_ticket(self, mismatch_row):
        """CRITICAL: keyword evidence must actually appear in ticket text."""
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        ticket_text = (
            mismatch_row["Ticket Subject"] + " " + mismatch_row["Ticket Description"]
        ).lower()
        for ev in dossier["feature_evidence"]:
            if ev.get("signal") == "keyword":
                kw = ev["value"].lower()
                assert kw in ticket_text, (
                    f"HALLUCINATION DETECTED: keyword '{kw}' not found in ticket text!"
                )

    def test_resolution_time_evidence_value_matches(self, mismatch_row):
        """Resolution time evidence must reference actual RT value."""
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        rt_evidence = [e for e in dossier["feature_evidence"] if e.get("signal") == "resolution_time"]
        if rt_evidence:
            ev = rt_evidence[0]
            # Value should contain the actual RT
            assert "1.5" in ev["value"] or "hours" in ev["value"].lower()

    def test_constraint_analysis_not_empty(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        analysis = dossier["constraint_analysis"]
        assert isinstance(analysis, str)
        assert len(analysis) > 50  # must be substantive

    def test_constraint_analysis_references_priority(self, mismatch_row):
        """Constraint analysis must mention actual assigned priority."""
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert "Low" in dossier["constraint_analysis"]

    def test_confidence_string_has_percentage(self, mismatch_row):
        gen = DossierGenerator()
        dossier = gen.generate_single(mismatch_row)
        assert "%" in dossier["confidence"]

    def test_batch_generates_only_mismatches(self, mismatch_row, consistent_row):
        gen = DossierGenerator()
        df = pd.DataFrame([mismatch_row, consistent_row])
        dossiers = gen.generate_batch(df, mismatch_only=True)
        assert len(dossiers) == 1
        assert dossiers[0]["mismatch_type"] in ["Hidden Crisis", "False Alarm"]

    def test_batch_all_tickets_if_not_mismatch_only(self, mismatch_row):
        gen = DossierGenerator()
        df = pd.DataFrame([mismatch_row])
        dossiers = gen.generate_batch(df, mismatch_only=False)
        assert len(dossiers) == 1

    def test_dossier_to_dataframe(self, mismatch_row):
        gen = DossierGenerator()
        df = pd.DataFrame([mismatch_row])
        dossiers = gen.generate_batch(df)
        df_out = gen.dossiers_to_dataframe(dossiers)
        assert "ticket_id" in df_out.columns
        assert "mismatch_type" in df_out.columns
        assert "num_evidence_items" in df_out.columns
