"""
baseline_model.py
=================
VERSION 1: TF-IDF + Logistic Regression Classifier.

Pipeline:
  Input: subject + description + channel + ticket_type
  Feature Engineering:
    - TF-IDF on combined text (1–3 grams)
    - Encoded metadata features (channel, ticket_type)
    - Normalized resolution time
  Classifier: Logistic Regression with class-weight balancing
  Output: binary label (Consistent=0, Mismatch=1)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from scipy.sparse import hstack, csr_matrix

from src.utils.config import (
    BASELINE_MODEL_DIR, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    LR_MAX_ITER, LR_C, RANDOM_SEED, COL_CHANNEL, COL_TICKET_TYPE,
    COL_RESOLUTION_TIME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineClassifier:
    """
    TF-IDF + Logistic Regression mismatch detector.

    Combines sparse TF-IDF features with dense metadata features.
    Handles class imbalance via class_weight='balanced'.

    Usage:
        clf = BaselineClassifier()
        clf.fit(X_train_df, y_train)
        preds = clf.predict(X_test_df)
        probs = clf.predict_proba(X_test_df)
        clf.save()
        clf2 = BaselineClassifier.load()
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            min_df=2,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.lr = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
            multi_class="auto",
        )
        self.channel_encoder = LabelEncoder()
        self.ticket_type_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._fitted = False

    def _build_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Builds combined feature matrix from text + metadata.

        Feature block 1: TF-IDF on combined_text (sparse, ~15k dims)
        Feature block 2: Metadata features (dense, 4 dims)
          - channel (label encoded)
          - ticket_type (label encoded)
          - resolution_time (log-normalized)
          - text_length (normalized)

        Args:
            df: Input DataFrame.
            fit: Whether to fit encoders/vectorizer (True for training).

        Returns:
            Combined feature matrix (sparse CSR).
        """
        # Text features
        if fit:
            tfidf_feats = self.tfidf.fit_transform(df["combined_text"])
        else:
            tfidf_feats = self.tfidf.transform(df["combined_text"])

        # Metadata features
        channels = df[COL_CHANNEL].fillna("Unknown").astype(str)
        ticket_types = df[COL_TICKET_TYPE].fillna("Unknown").astype(str)
        res_times = np.log1p(df[COL_RESOLUTION_TIME].fillna(24).astype(float)).values.reshape(-1, 1)
        text_len = (df["combined_text"].str.len() / 1000.0).values.reshape(-1, 1)

        if fit:
            ch_enc = self.channel_encoder.fit_transform(channels).reshape(-1, 1)
            tt_enc = self.ticket_type_encoder.fit_transform(ticket_types).reshape(-1, 1)
            meta = np.hstack([ch_enc, tt_enc, res_times, text_len])
            meta = self.scaler.fit_transform(meta)
        else:
            # Handle unseen labels gracefully
            ch_safe = channels.map(
                lambda x: x if x in self.channel_encoder.classes_ else self.channel_encoder.classes_[0]
            )
            tt_safe = ticket_types.map(
                lambda x: x if x in self.ticket_type_encoder.classes_ else self.ticket_type_encoder.classes_[0]
            )
            ch_enc = self.channel_encoder.transform(ch_safe).reshape(-1, 1)
            tt_enc = self.ticket_type_encoder.transform(tt_safe).reshape(-1, 1)
            meta = np.hstack([ch_enc, tt_enc, res_times, text_len])
            meta = self.scaler.transform(meta)

        # Combine sparse TF-IDF with dense metadata
        meta_sparse = csr_matrix(meta)
        combined = hstack([tfidf_feats, meta_sparse])
        return combined

    def fit(self, df: pd.DataFrame, labels: pd.Series) -> "BaselineClassifier":
        """
        Fits TF-IDF vectorizer and Logistic Regression classifier.

        Args:
            df: Training DataFrame.
            labels: Binary labels (0=Consistent, 1=Mismatch).

        Returns:
            self (for chaining).
        """
        logger.info(f"Fitting BaselineClassifier on {len(df)} samples...")
        logger.info(f"Class distribution: {labels.value_counts().to_dict()}")

        X = self._build_features(df, fit=True)
        self.lr.fit(X, labels)
        self._fitted = True

        logger.info("Baseline model fitted successfully.")
        logger.info(f"TF-IDF vocabulary size: {len(self.tfidf.vocabulary_)}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts binary mismatch labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._build_features(df, fit=False)
        return self.lr.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities. Column 1 = P(Mismatch)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._build_features(df, fit=False)
        return self.lr.predict_proba(X)

    def predict_single(self, subject: str, description: str,
                        channel: str = "Email", ticket_type: str = "Technical Issue",
                        resolution_time: float = 24.0) -> Dict:
        """
        Predicts for a single ticket (for Streamlit app).

        Args:
            subject: Ticket subject.
            description: Ticket description.
            channel: Intake channel.
            ticket_type: Type of ticket.
            resolution_time: Hours to resolve.

        Returns:
            Dict with prediction, confidence, label.
        """
        import re
        combined = f"{subject.lower()} {description.lower()}".strip()
        row = {
            "combined_text": combined,
            COL_CHANNEL: channel,
            COL_TICKET_TYPE: ticket_type,
            COL_RESOLUTION_TIME: resolution_time,
        }
        single_df = pd.DataFrame([row])
        prob = self.predict_proba(single_df)[0]
        pred = int(prob[1] >= 0.5)
        return {
            "prediction": pred,
            "label": "Mismatch" if pred == 1 else "Consistent",
            "confidence": round(float(prob[1]) if pred == 1 else float(prob[0]), 4),
            "mismatch_probability": round(float(prob[1]), 4),
        }

    def save(self, path: Path = None) -> Path:
        """Saves model artifacts to disk."""
        save_dir = path or BASELINE_MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "tfidf": self.tfidf,
            "lr": self.lr,
            "channel_encoder": self.channel_encoder,
            "ticket_type_encoder": self.ticket_type_encoder,
            "scaler": self.scaler,
        }
        save_path = save_dir / "baseline_model.joblib"
        joblib.dump(artifact, save_path)
        logger.info(f"Baseline model saved to: {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Path = None) -> "BaselineClassifier":
        """Loads a saved model from disk."""
        load_path = path or (BASELINE_MODEL_DIR / "baseline_model.joblib")
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model at {load_path}")
        artifact = joblib.load(load_path)
        clf = cls()
        clf.tfidf = artifact["tfidf"]
        clf.lr = artifact["lr"]
        clf.channel_encoder = artifact["channel_encoder"]
        clf.ticket_type_encoder = artifact["ticket_type_encoder"]
        clf.scaler = artifact["scaler"]
        clf._fitted = True
        logger.info(f"Baseline model loaded from: {load_path}")
        return clf
