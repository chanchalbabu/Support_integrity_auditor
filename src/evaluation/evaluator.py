"""
evaluator.py
============
Complete evaluation system for SIA.

Metrics computed:
  - Binary Classification Accuracy
  - Macro F1 Score
  - Per-Class Precision, Recall, F1
  - Confusion Matrix
  - Pseudo-Label Signal Agreement
  - ROC-AUC (if probabilities available)

Targets:
  Accuracy  >= 83%
  Macro F1  >= 0.82
  Per-Class Recall >= 0.78 (both classes)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    ConfusionMatrixDisplay,
)

from src.utils.config import (
    MIN_ACCURACY, MIN_MACRO_F1, MIN_PER_CLASS_RECALL, REPORTS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_NAMES = ["Consistent", "Mismatch"]


class SIAEvaluator:
    """
    Evaluates model performance against MARS Open Projects 2026 thresholds.

    Usage:
        evaluator = SIAEvaluator()
        results = evaluator.evaluate(y_true, y_pred, y_proba)
        evaluator.print_report(results)
        evaluator.save_report(results)
        evaluator.plot_confusion_matrix(y_true, y_pred)
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> Dict:
        """
        Computes all evaluation metrics.

        Args:
            y_true: Ground-truth binary labels (0=Consistent, 1=Mismatch).
            y_pred: Predicted binary labels.
            y_proba: Optional predicted probabilities (column 1 = P(Mismatch)).
            model_name: Name for the report.

        Returns:
            Dict with all metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        per_class_precision = precision_score(y_true, y_pred, average=None, labels=[0, 1])
        per_class_recall = recall_score(y_true, y_pred, average=None, labels=[0, 1])
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "n_samples": int(len(y_true)),
            "n_mismatch": int(y_true.sum()),
            "mismatch_rate": round(float(y_true.mean()), 4),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                "Consistent": {
                    "precision": round(float(per_class_precision[0]), 4),
                    "recall": round(float(per_class_recall[0]), 4),
                    "f1": round(float(per_class_f1[0]), 4),
                    "support": int((y_true == 0).sum()),
                },
                "Mismatch": {
                    "precision": round(float(per_class_precision[1]), 4),
                    "recall": round(float(per_class_recall[1]), 4),
                    "f1": round(float(per_class_f1[1]), 4),
                    "support": int((y_true == 1).sum()),
                },
            },
            "confusion_matrix": cm.tolist(),
        }

        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                results["roc_auc"] = round(roc_auc, 4)
            except Exception:
                results["roc_auc"] = None

        # Threshold checks
        results["thresholds"] = {
            "accuracy_pass": accuracy >= MIN_ACCURACY,
            "macro_f1_pass": macro_f1 >= MIN_MACRO_F1,
            "consistent_recall_pass": float(per_class_recall[0]) >= MIN_PER_CLASS_RECALL,
            "mismatch_recall_pass": float(per_class_recall[1]) >= MIN_PER_CLASS_RECALL,
        }
        results["overall_pass"] = all(results["thresholds"].values())

        return results

    def compute_signal_agreement(self, df: pd.DataFrame) -> Dict:
        """
        Computes pairwise signal agreement for pseudo-label evaluation.

        Args:
            df: DataFrame with signal severity columns.

        Returns:
            Dict of pairwise agreement rates.
        """
        signal_cols = {
            "semantic": "semantic_severity",
            "resolution_time": "rt_severity",
            "rule_based": "rule_severity",
        }
        existing = {k: v for k, v in signal_cols.items() if v in df.columns}
        pairs = {}
        keys = list(existing.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                agreement = (df[existing[a]] == df[existing[b]]).mean()
                pairs[f"{a}_vs_{b}"] = round(float(agreement), 4)

        # Each signal vs fused
        if "inferred_severity" in df.columns:
            for k, col in existing.items():
                agreement = (df[col] == df["inferred_severity"]).mean()
                pairs[f"{k}_vs_fused"] = round(float(agreement), 4)

        return pairs

    def print_report(self, results: Dict) -> None:
        """Prints a formatted evaluation report to stdout."""
        line = "=" * 60
        logger.info(line)
        logger.info(f"EVALUATION REPORT — {results['model_name']}")
        logger.info(line)
        logger.info(f"Samples:       {results['n_samples']}")
        logger.info(f"Mismatches:    {results['n_mismatch']} ({results['mismatch_rate']*100:.1f}%)")
        logger.info("")
        logger.info(f"Accuracy:      {results['accuracy']*100:.2f}%  (target ≥ {MIN_ACCURACY*100:.0f}%) {'✓' if results['thresholds']['accuracy_pass'] else '✗'}")
        logger.info(f"Macro F1:      {results['macro_f1']:.4f}    (target ≥ {MIN_MACRO_F1:.2f})  {'✓' if results['thresholds']['macro_f1_pass'] else '✗'}")
        if "roc_auc" in results:
            logger.info(f"ROC-AUC:       {results['roc_auc']:.4f}")
        logger.info("")
        logger.info("Per-Class Metrics:")
        for cls_name, metrics in results["per_class"].items():
            recall_pass = results["thresholds"].get(f"{cls_name.lower()}_recall_pass", True)
            logger.info(
                f"  {cls_name:12s}: P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f} {'✓' if recall_pass else '✗'}  "
                f"F1={metrics['f1']:.3f}  n={metrics['support']}"
            )
        logger.info("")
        status = "✓ ALL THRESHOLDS MET" if results["overall_pass"] else "✗ SOME THRESHOLDS FAILED"
        logger.info(f"Status: {status}")
        logger.info(line)

    def save_report(self, results: Dict, filename: str = None) -> Path:
        """Saves evaluation results as JSON."""
        fname = filename or f"eval_{results['model_name'].lower().replace(' ', '_')}.json"
        path = self.output_dir / fname
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation report saved to: {path}")
        return path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save: bool = True,
    ) -> plt.Figure:
        """Plots and optionally saves a styled confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            path = self.output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to: {path}")

        return fig

    def plot_metrics_comparison(
        self,
        results_list: List[Dict],
        save: bool = True,
    ) -> plt.Figure:
        """
        Plots side-by-side metric comparison for multiple models.

        Args:
            results_list: List of results dicts from evaluate().
            save: Whether to save the figure.

        Returns:
            Matplotlib Figure.
        """
        models = [r["model_name"] for r in results_list]
        metrics = {
            "Accuracy": [r["accuracy"] for r in results_list],
            "Macro F1": [r["macro_f1"] for r in results_list],
            "Consistent Recall": [r["per_class"]["Consistent"]["recall"] for r in results_list],
            "Mismatch Recall": [r["per_class"]["Mismatch"]["recall"] for r in results_list],
        }
        targets = {
            "Accuracy": MIN_ACCURACY,
            "Macro F1": MIN_MACRO_F1,
            "Consistent Recall": MIN_PER_CLASS_RECALL,
            "Mismatch Recall": MIN_PER_CLASS_RECALL,
        }

        x = np.arange(len(models))
        width = 0.18
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

        for i, (metric, values) in enumerate(metrics.items()):
            bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)
            ax.axhline(y=targets[metric], color=colors[i], linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save:
            path = self.output_dir / "model_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to: {path}")

        return fig
