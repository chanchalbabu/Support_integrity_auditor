"""
train_pipeline.py
=================
Standalone end-to-end training pipeline for SIA.

Steps:
  1. Load and clean dataset
  2. Generate pseudo-labels (3 signals + fusion)
  3. Split into train/val/test
  4. Train Baseline (TF-IDF + LR)
  5. Train Advanced (DeBERTa-v3-small) [optional]
  6. Evaluate both models
  7. Save models, labels, and evaluation reports
  8. Generate dossiers for test set

Usage:
  python train_pipeline.py
  python train_pipeline.py --skip-advanced
  python train_pipeline.py --data-path data/raw/my_dataset.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import (
    PROCESSED_DATA_DIR, PSEUDO_LABEL_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE,
)
from src.utils.data_loader import load_dataset
from src.utils.logger import get_logger
from src.pseudo_labeling.fusion import PseudoLabelGenerator
from src.ml.baseline_model import BaselineClassifier
from src.evaluation.evaluator import SIAEvaluator
from src.dossier.generator import DossierGenerator

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SIA Training Pipeline")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to CSV dataset (default: uses synthetic data)")
    parser.add_argument("--skip-advanced", action="store_true",
                        help="Skip DeBERTa training (faster, baseline only)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("  SUPPORT INTEGRITY AUDITOR (SIA) — TRAINING PIPELINE")
    logger.info("  MARS Open Projects 2026")
    logger.info("=" * 70)

    # ── STEP 1: Load Dataset ──────────────────────────────────────────
    logger.info("\n[STEP 1/7] Loading dataset...")
    data_path = Path(args.data_path) if args.data_path else None
    df = load_dataset(path=data_path, synthetic_fallback=True)
    logger.info(f"Dataset shape: {df.shape}")

    # Save cleaned dataset
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "cleaned_tickets.csv", index=False)

    # ── STEP 2: Pseudo-Label Generation ──────────────────────────────
    logger.info("\n[STEP 2/7] Generating pseudo-labels...")
    plg = PseudoLabelGenerator()
    df_labeled = plg.generate(df)

    # Ablation study
    ablation_df = plg.get_ablation_stats(df_labeled)
    contribution_df = plg.get_signal_contribution(df_labeled)
    logger.info(f"\nSignal Contribution:\n{contribution_df.to_string(index=False)}")

    # Save pseudo-labels
    PSEUDO_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(PSEUDO_LABEL_DIR / "pseudo_labeled_tickets.csv", index=False)
    ablation_df.to_csv(PSEUDO_LABEL_DIR / "ablation_study.csv", index=False)
    contribution_df.to_csv(PSEUDO_LABEL_DIR / "signal_contribution.csv", index=False)

    logger.info(f"\nLabel distribution:\n{df_labeled['mismatch_label'].value_counts().to_string()}")

    # ── STEP 3: Train/Val/Test Split ──────────────────────────────────
    logger.info("\n[STEP 3/7] Splitting data...")
    labels = df_labeled["label"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df_labeled, labels,
        test_size=TEST_SIZE,
        random_state=args.seed,
        stratify=labels,
    )
    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative_size,
        random_state=args.seed,
        stratify=y_trainval,
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info(f"Train mismatch rate: {y_train.mean():.3f}")
    logger.info(f"Test mismatch rate:  {y_test.mean():.3f}")

    evaluator = SIAEvaluator()
    results_all = []

    # ── STEP 4: Baseline Model ────────────────────────────────────────
    logger.info("\n[STEP 4/7] Training Baseline (TF-IDF + LR)...")
    baseline = BaselineClassifier()
    baseline.fit(X_train, y_train)
    baseline.save()

    y_pred_base = baseline.predict(X_test)
    y_proba_base = baseline.predict_proba(X_test)
    results_base = evaluator.evaluate(
        y_test.values, y_pred_base, y_proba_base, model_name="Baseline TF-IDF+LR"
    )
    evaluator.print_report(results_base)
    evaluator.save_report(results_base)
    evaluator.plot_confusion_matrix(y_test.values, y_pred_base, "Baseline TF-IDF+LR")
    results_all.append(results_base)

    # ── STEP 5: Advanced Model (DeBERTa) ─────────────────────────────
    if not args.skip_advanced:
        logger.info("\n[STEP 5/7] Training Advanced Model (DeBERTa-v3-small)...")
        try:
            from src.ml.advanced_model import AdvancedModelTrainer
            trainer = AdvancedModelTrainer()
            history = trainer.train(X_train, y_train, X_val, y_val)
            trainer.save()

            y_pred_adv, y_proba_adv = trainer.predict(X_test)
            results_adv = evaluator.evaluate(
                y_test.values, y_pred_adv, y_proba_adv, model_name="DeBERTa-v3-small"
            )
            evaluator.print_report(results_adv)
            evaluator.save_report(results_adv)
            evaluator.plot_confusion_matrix(y_test.values, y_pred_adv, "DeBERTa-v3-small")
            results_all.append(results_adv)

            # Training history
            history_path = Path("outputs/reports/training_history.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Advanced model training failed: {e}")
            logger.warning("Continuing with baseline model only.")
    else:
        logger.info("\n[STEP 5/7] Skipping advanced model (--skip-advanced flag set).")

    # ── STEP 6: Model Comparison ──────────────────────────────────────
    logger.info("\n[STEP 6/7] Generating model comparison...")
    if len(results_all) > 1:
        evaluator.plot_metrics_comparison(results_all)

    # Signal agreement stats
    signal_agreement = evaluator.compute_signal_agreement(df_labeled)
    logger.info(f"\nSignal Agreement:\n{json.dumps(signal_agreement, indent=2)}")
    with open(Path("outputs/reports/signal_agreement.json"), "w") as f:
        json.dump(signal_agreement, f, indent=2)

    # ── STEP 7: Dossier Generation ────────────────────────────────────
    logger.info("\n[STEP 7/7] Generating evidence dossiers for test set mismatches...")

    # Use baseline predictions (or advanced if available) on test set
    X_test_with_preds = X_test.copy()
    if not args.skip_advanced and len(results_all) > 1:
        X_test_with_preds["mismatch_label"] = np.where(y_pred_adv == 1, "Mismatch", "Consistent")
    else:
        X_test_with_preds["mismatch_label"] = np.where(y_pred_base == 1, "Mismatch", "Consistent")

    dossier_gen = DossierGenerator()
    dossiers = dossier_gen.generate_batch(X_test_with_preds)
    dossier_gen.save_dossiers(dossiers, filename="test_set_dossiers.json")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Models saved:    models/")
    logger.info(f"  Pseudo-labels:   data/pseudo_labels/")
    logger.info(f"  Dossiers:        outputs/dossiers/")
    logger.info(f"  Reports:         outputs/reports/")
    logger.info(f"  Mismatch rate:   {df_labeled['label'].mean()*100:.1f}%")
    logger.info(f"  Baseline Acc:    {results_base['accuracy']*100:.1f}%")
    logger.info(f"  Baseline F1:     {results_base['macro_f1']:.4f}")
    status = "✓ SUBMISSION READY" if results_base["overall_pass"] else "⚠ BELOW THRESHOLD"
    logger.info(f"  Status:          {status}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
