"""
test_pipeline.py
================
Integration tests for the complete SIA pipeline.
Tests end-to-end flow from data loading → pseudo-labels → classifier → dossier.
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import generate_synthetic_dataset, validate_and_clean
from src.pseudo_labeling.fusion import PseudoLabelGenerator
from src.ml.baseline_model import BaselineClassifier
from src.evaluation.evaluator import SIAEvaluator
from src.dossier.generator import DossierGenerator
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def pipeline_data():
    """Generates and processes a small dataset once for all integration tests."""
    df_raw = generate_synthetic_dataset(n_tickets=300, seed=42)
    df = validate_and_clean(df_raw)

    plg = PseudoLabelGenerator()
    df_labeled = plg.generate(df)

    labels = df_labeled["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        df_labeled, labels, test_size=0.3, random_state=42, stratify=labels
    )

    clf = BaselineClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    return {
        "df_labeled": df_labeled,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "clf": clf,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


class TestDataPipeline:

    def test_synthetic_data_has_required_columns(self):
        df = generate_synthetic_dataset(n_tickets=50)
        df = validate_and_clean(df)
        assert "Ticket ID" in df.columns
        assert "Ticket Subject" in df.columns
        assert "Ticket Priority" in df.columns
        assert "combined_text" in df.columns

    def test_cleaning_removes_invalid_priorities(self):
        df = generate_synthetic_dataset(n_tickets=100)
        df = validate_and_clean(df)
        valid = {"Low", "Medium", "High", "Critical"}
        assert set(df["Ticket Priority"].unique()).issubset(valid)

    def test_resolution_time_is_numeric(self):
        df = generate_synthetic_dataset(n_tickets=50)
        df = validate_and_clean(df)
        assert df["Resolution Time"].dtype in [np.float64, np.float32, float]
        assert (df["Resolution Time"] > 0).all()


class TestPseudoLabelPipeline:

    def test_all_label_columns_present(self, pipeline_data):
        df = pipeline_data["df_labeled"]
        for col in ["inferred_severity", "mismatch_label", "mismatch_type", "label"]:
            assert col in df.columns

    def test_mismatch_rate_reasonable(self, pipeline_data):
        df = pipeline_data["df_labeled"]
        rate = df["label"].mean()
        # Mismatch rate should be between 10% and 90% for a synthetic dataset
        assert 0.05 <= rate <= 0.95, f"Mismatch rate {rate:.2%} seems unreasonable"

    def test_inferred_severity_valid_values(self, pipeline_data):
        df = pipeline_data["df_labeled"]
        valid = {"Low", "Medium", "High", "Critical"}
        assert set(df["inferred_severity"].unique()).issubset(valid)


class TestClassifierPipeline:

    def test_predictions_are_binary(self, pipeline_data):
        y_pred = pipeline_data["y_pred"]
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_probabilities_sum_to_one(self, pipeline_data):
        y_proba = pipeline_data["y_proba"]
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_in_range(self, pipeline_data):
        y_proba = pipeline_data["y_proba"]
        assert (y_proba >= 0).all() and (y_proba <= 1).all()

    def test_accuracy_above_baseline(self, pipeline_data):
        """Classifier must beat random chance (50%) by a wide margin."""
        y_true = pipeline_data["y_test"].values
        y_pred = pipeline_data["y_pred"]
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)
        assert acc >= 0.65, f"Accuracy {acc:.2%} — too low even for synthetic data"

    def test_single_ticket_inference_returns_dict(self, pipeline_data):
        clf = pipeline_data["clf"]
        result = clf.predict_single(
            subject="Server is completely down",
            description="All users affected, revenue impact confirmed",
            channel="Email",
            ticket_type="Technical Issue",
            resolution_time=2.0,
        )
        assert "prediction" in result
        assert "label" in result
        assert "confidence" in result
        assert "mismatch_probability" in result
        assert result["label"] in ["Mismatch", "Consistent"]
        assert 0.0 <= result["confidence"] <= 1.0


class TestEvaluatorPipeline:

    def test_evaluator_returns_all_metrics(self, pipeline_data):
        evaluator = SIAEvaluator()
        results = evaluator.evaluate(
            pipeline_data["y_test"].values,
            pipeline_data["y_pred"],
            pipeline_data["y_proba"],
            model_name="Integration Test",
        )
        required = ["accuracy", "macro_f1", "per_class", "confusion_matrix", "thresholds"]
        for key in required:
            assert key in results

    def test_per_class_has_both_classes(self, pipeline_data):
        evaluator = SIAEvaluator()
        results = evaluator.evaluate(
            pipeline_data["y_test"].values,
            pipeline_data["y_pred"],
            model_name="Test",
        )
        assert "Consistent" in results["per_class"]
        assert "Mismatch" in results["per_class"]

    def test_signal_agreement_computed(self, pipeline_data):
        evaluator = SIAEvaluator()
        agreement = evaluator.compute_signal_agreement(pipeline_data["df_labeled"])
        assert isinstance(agreement, dict)
        assert len(agreement) > 0


class TestDossierPipeline:

    def test_dossiers_generated_for_mismatches(self, pipeline_data):
        df = pipeline_data["X_test"].copy()
        y_pred = pipeline_data["y_pred"]
        df["mismatch_label"] = np.where(y_pred == 1, "Mismatch", "Consistent")
        df["mismatch_type"] = df.apply(
            lambda r: r.get("mismatch_type", "Hidden Crisis") if r["mismatch_label"] == "Mismatch" else "",
            axis=1,
        )

        gen = DossierGenerator()
        dossiers = gen.generate_batch(df)
        n_mismatches = (y_pred == 1).sum()
        assert len(dossiers) == n_mismatches

    def test_no_hallucination_in_dossiers(self, pipeline_data):
        """Every keyword in every dossier must exist in the ticket text."""
        df = pipeline_data["X_test"].copy()
        y_pred = pipeline_data["y_pred"]
        df["mismatch_label"] = np.where(y_pred == 1, "Mismatch", "Consistent")
        # Fill required columns
        for col in ["mismatch_type", "inferred_severity", "fusion_confidence",
                    "signal_agreement", "rt_severity", "rt_score", "semantic_severity", "semantic_score"]:
            if col not in df.columns:
                df[col] = "Medium" if "severity" in col else 0.7

        gen = DossierGenerator()
        dossiers = gen.generate_batch(df)

        for dossier in dossiers:
            ticket_text = ""
            orig_row = df[df["Ticket ID"] == dossier["ticket_id"]]
            if not orig_row.empty:
                row = orig_row.iloc[0]
                ticket_text = (
                    str(row.get("Ticket Subject", "")) + " " +
                    str(row.get("Ticket Description", ""))
                ).lower()

            for ev in dossier.get("feature_evidence", []):
                if ev.get("signal") == "keyword" and ticket_text:
                    kw = ev["value"].lower()
                    assert kw in ticket_text, (
                        f"HALLUCINATION: '{kw}' not in ticket {dossier['ticket_id']}"
                    )
