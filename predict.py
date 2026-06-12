"""
predict.py
==========
Inference script for SIA.

Accepts a CSV file with ticket data, runs the trained model,
and outputs predictions + evidence dossiers.

Usage:
  python predict.py --input data/raw/new_tickets.csv
  python predict.py --input tickets.csv --model advanced --output-dir outputs/
  python predict.py --subject "Server is down" --description "All users affected" --priority Low
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import (
    BASELINE_MODEL_DIR, ADVANCED_MODEL_DIR,
    COL_SUBJECT, COL_DESCRIPTION, COL_PRIORITY, COL_CHANNEL,
    COL_TICKET_TYPE, COL_RESOLUTION_TIME,
)
from src.utils.data_loader import validate_and_clean
from src.utils.logger import get_logger
from src.pseudo_labeling.fusion import create_mismatch_label
from src.pseudo_labeling.signal_rule_based import RuleBasedSeverityScorer
from src.pseudo_labeling.signal_resolution_time import ResolutionTimeSeverityScorer
from src.dossier.generator import DossierGenerator

logger = get_logger(__name__)


def load_model(model_type: str = "baseline"):
    """Loads the appropriate trained model."""
    if model_type == "advanced":
        try:
            from src.ml.advanced_model import AdvancedModelTrainer
            return AdvancedModelTrainer.load(), "advanced"
        except Exception as e:
            logger.warning(f"Advanced model load failed ({e}). Falling back to baseline.")
    from src.ml.baseline_model import BaselineClassifier
    return BaselineClassifier.load(), "baseline"


def infer_severity_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast severity inference using rule-based + resolution-time signals
    (no GPU/embedding required for predict.py speed).
    """
    rule_scorer = RuleBasedSeverityScorer()
    rt_scorer = ResolutionTimeSeverityScorer()

    df = rule_scorer.score(df)
    rt_scorer.fit(df)
    df = rt_scorer.score(df)

    # Use rule_severity as primary inferred severity for speed
    df["inferred_severity"] = df["rule_severity"]
    df["fusion_confidence"] = df["rule_score"]
    df["signal_agreement"] = 0.67  # 2/3 signals agree by default

    mismatch_results = [
        create_mismatch_label(row[COL_PRIORITY], row["inferred_severity"])
        for _, row in df.iterrows()
    ]
    df["mismatch_label"] = [r[0] for r in mismatch_results]
    df["mismatch_type"] = [r[1] for r in mismatch_results]
    df["severity_delta"] = [r[2] for r in mismatch_results]
    df["label"] = (df["mismatch_label"] == "Mismatch").astype(int)

    # Placeholders for dossier
    df["semantic_severity"] = df["rule_severity"]
    df["semantic_score"] = df["rule_score"]

    return df


def predict_csv(
    input_path: str,
    model_type: str = "baseline",
    output_dir: str = "outputs/predictions",
) -> pd.DataFrame:
    """
    Runs inference on a CSV file.

    Args:
        input_path: Path to input CSV.
        model_type: "baseline" or "advanced".
        output_dir: Directory for output files.

    Returns:
        DataFrame with predictions and dossiers.
    """
    logger.info(f"Loading input: {input_path}")
    df_raw = pd.read_csv(input_path)
    df = validate_and_clean(df_raw)

    # Infer severity for dossier generation
    df = infer_severity_fast(df)

    # Load model and predict
    model, loaded_type = load_model(model_type)
    logger.info(f"Using {loaded_type} model for prediction...")

    if loaded_type == "baseline":
        probs = model.predict_proba(df)
        preds = (probs[:, 1] >= 0.5).astype(int)
    else:
        preds, probs = model.predict(df)

    df["predicted_label"] = np.where(preds == 1, "Mismatch", "Consistent")
    df["mismatch_probability"] = probs[:, 1] if probs.ndim == 2 else probs

    # Override mismatch_label with model prediction
    df["mismatch_label"] = df["predicted_label"]

    # Generate dossiers
    dossier_gen = DossierGenerator()
    dossiers = dossier_gen.generate_batch(df)

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictions_path = out / "predictions.csv"
    df[[
        "Ticket ID", COL_SUBJECT, COL_PRIORITY, "inferred_severity",
        "predicted_label", "mismatch_probability", "mismatch_type", "severity_delta",
    ]].to_csv(predictions_path, index=False)

    dossier_path = out / "dossiers.json"
    dossier_gen.save_dossiers(dossiers, output_dir=out)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  Predictions: {predictions_path}")
    logger.info(f"  Dossiers:    {dossier_path}")
    logger.info(f"  Total:       {len(df)} tickets")
    logger.info(f"  Mismatches:  {(preds==1).sum()} ({(preds==1).mean()*100:.1f}%)")

    return df


def predict_single_ticket(
    subject: str,
    description: str,
    priority: str,
    channel: str = "Email",
    ticket_type: str = "Technical Issue",
    resolution_time: float = 24.0,
    model_type: str = "baseline",
) -> dict:
    """
    Predicts for a single ticket (used by Streamlit app).

    Returns:
        Dict with full prediction + dossier.
    """
    row_data = {
        "Ticket ID": "ADHOC-001",
        COL_SUBJECT: subject,
        COL_DESCRIPTION: description,
        COL_PRIORITY: priority,
        COL_CHANNEL: channel,
        COL_TICKET_TYPE: ticket_type,
        COL_RESOLUTION_TIME: resolution_time,
        "combined_text": f"{subject.lower()} {description.lower()}",
        "Customer Email": "user@example.com",
        "Product Purchased": "Unknown",
    }
    df = pd.DataFrame([row_data])

    # Infer severity
    rule_scorer = RuleBasedSeverityScorer()
    df = rule_scorer.score(df)
    df["inferred_severity"] = df["rule_severity"]
    df["fusion_confidence"] = df["rule_score"]
    df["signal_agreement"] = 0.67
    df["semantic_severity"] = df["rule_severity"]
    df["semantic_score"] = df["rule_score"]

    mismatch_label, mismatch_type, severity_delta = create_mismatch_label(
        priority, df["rule_severity"].iloc[0]
    )
    df["mismatch_label"] = mismatch_label
    df["mismatch_type"] = mismatch_type
    df["severity_delta"] = severity_delta

    # Model prediction
    try:
        model, loaded_type = load_model(model_type)
        if loaded_type == "baseline":
            result = model.predict_single(subject, description, channel, ticket_type, resolution_time)
        else:
            result = model.predict_single(subject, description, channel, ticket_type, resolution_time)
    except Exception:
        prob = df["rule_score"].iloc[0] if mismatch_label == "Mismatch" else 1 - df["rule_score"].iloc[0]
        result = {
            "label": mismatch_label,
            "confidence": float(prob),
            "mismatch_probability": float(df["rule_score"].iloc[0]),
        }

    # Generate dossier
    dossier_gen = DossierGenerator()
    dossier = dossier_gen.generate_single(df.iloc[0])

    return {
        "prediction": result,
        "inferred_severity": df["inferred_severity"].iloc[0],
        "mismatch_label": mismatch_label,
        "mismatch_type": mismatch_type,
        "severity_delta": severity_delta,
        "dossier": dossier,
    }


def main():
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    subparsers = parser.add_subparsers(dest="mode")

    # CSV mode
    csv_parser = subparsers.add_parser("csv", help="Predict on a CSV file")
    csv_parser.add_argument("--input", required=True, help="Input CSV path")
    csv_parser.add_argument("--model", default="baseline", choices=["baseline", "advanced"])
    csv_parser.add_argument("--output-dir", default="outputs/predictions")

    # Single ticket mode
    single_parser = subparsers.add_parser("single", help="Predict on a single ticket")
    single_parser.add_argument("--subject", required=True)
    single_parser.add_argument("--description", required=True)
    single_parser.add_argument("--priority", required=True, choices=["Low", "Medium", "High", "Critical"])
    single_parser.add_argument("--channel", default="Email")
    single_parser.add_argument("--ticket-type", default="Technical Issue")
    single_parser.add_argument("--resolution-time", type=float, default=24.0)
    single_parser.add_argument("--model", default="baseline", choices=["baseline", "advanced"])

    args = parser.parse_args()

    if args.mode == "csv":
        predict_csv(args.input, args.model, args.output_dir)
    elif args.mode == "single":
        result = predict_single_ticket(
            args.subject, args.description, args.priority,
            args.channel, args.ticket_type, args.resolution_time, args.model,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
