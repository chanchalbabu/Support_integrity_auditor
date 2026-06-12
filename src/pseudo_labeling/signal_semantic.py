"""
signal_semantic.py
==================
Signal A: Semantic Severity Scoring using Sentence Transformers.

Approach:
  - Embed each ticket's combined text using all-MiniLM-L6-v2.
  - Embed 4 anchor sentences representing each severity level.
  - Assign severity based on cosine similarity to anchors.
  - Normalize scores to [0, 1] range per severity level.

Output:
  - semantic_severity: "Low" | "Medium" | "High" | "Critical"
  - semantic_score:    float in [0, 1] (confidence)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from src.utils.config import SBERT_MODEL, PRIORITY_LEVELS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# SEVERITY ANCHOR SENTENCES
# Reference descriptions for each severity level.
# Multiple anchors per level → averaged embedding.
# ─────────────────────────────────────────────
SEVERITY_ANCHORS = {
    "Low": [
        "I want to change my profile picture.",
        "Please update my notification preferences.",
        "I have a minor suggestion for the user interface.",
        "Can you help me find the documentation?",
        "I would like to update my account settings.",
    ],
    "Medium": [
        "I cannot access a specific feature in the application.",
        "The report is generating incorrect data for some entries.",
        "I am experiencing intermittent login issues.",
        "Some users are unable to see the updated content.",
        "The API is returning unexpected results occasionally.",
    ],
    "High": [
        "The application is crashing repeatedly for multiple users.",
        "Authentication is broken and users cannot log in.",
        "The integration with our payment system is failing.",
        "Performance is severely degraded and the service is very slow.",
        "A major feature is completely broken affecting our workflow.",
    ],
    "Critical": [
        "Our production server is completely down and no one can access the system.",
        "The payment gateway has failed and customers cannot complete purchases.",
        "We have detected a security breach and customer data may be compromised.",
        "Complete service outage affecting all users with severe revenue impact.",
        "Database corruption detected and data loss is occurring in production.",
    ],
}


class SemanticSeverityScorer:
    """
    Computes semantic severity scores using sentence embeddings.

    Usage:
        scorer = SemanticSeverityScorer()
        df = scorer.score(df)
    """

    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name
        self._model = None
        self._anchor_embeddings = None

    def _load_model(self):
        """Lazy-loads the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._anchor_embeddings = self._compute_anchor_embeddings()
                logger.info("SentenceTransformer loaded successfully.")
            except ImportError:
                logger.warning("sentence-transformers not installed. Using TF-IDF fallback.")
                self._model = "fallback"

    def _compute_anchor_embeddings(self) -> dict:
        """Computes and caches averaged anchor embeddings per severity level."""
        anchors = {}
        for level, sentences in SEVERITY_ANCHORS.items():
            embs = self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            anchors[level] = embs.mean(axis=0)
            logger.debug(f"Anchor embedding for '{level}': shape={anchors[level].shape}")
        return anchors

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def score_texts(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Scores a list of texts semantically.

        Args:
            texts: List of combined ticket texts.

        Returns:
            Tuple of (severity_labels, confidence_scores).
        """
        self._load_model()

        if self._model == "fallback":
            return self._tfidf_fallback(texts)

        logger.info(f"Computing semantic embeddings for {len(texts)} tickets...")
        ticket_embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64
        )

        severities = []
        confidences = []

        for emb in ticket_embeddings:
            sims = {
                level: self._cosine_similarity(emb, anchor_emb)
                for level, anchor_emb in self._anchor_embeddings.items()
            }
            best_level = max(sims, key=sims.get)
            # Normalize confidence: similarity relative to max possible
            sim_values = np.array(list(sims.values()))
            # Softmax over similarities for confidence
            exp_sims = np.exp(sim_values * 10)  # scale factor for sharper distribution
            confidence = float(exp_sims.max() / exp_sims.sum())
            severities.append(best_level)
            confidences.append(round(confidence, 4))

        return severities, confidences

    def _tfidf_fallback(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        TF-IDF fallback when sentence-transformers is unavailable.
        Uses keyword matching against anchor sentences.
        """
        logger.warning("Using TF-IDF keyword fallback for semantic scoring.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        anchor_docs = {
            level: " ".join(sents) for level, sents in SEVERITY_ANCHORS.items()
        }
        all_docs = list(anchor_docs.values()) + texts
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        anchor_matrix = tfidf_matrix[: len(anchor_docs)]
        ticket_matrix = tfidf_matrix[len(anchor_docs) :]

        sim_matrix = cosine_similarity(ticket_matrix, anchor_matrix)
        levels = list(anchor_docs.keys())

        severities = [levels[row.argmax()] for row in sim_matrix]
        confidences = [float(row.max()) for row in sim_matrix]
        return severities, confidences

    def score(self, df: pd.DataFrame, text_col: str = "combined_text") -> pd.DataFrame:
        """
        Adds semantic severity columns to the DataFrame.

        Args:
            df: Input DataFrame with text column.
            text_col: Column containing combined ticket text.

        Returns:
            DataFrame with added columns:
              - semantic_severity
              - semantic_score
        """
        texts = df[text_col].tolist()
        severities, confidences = self.score_texts(texts)
        df = df.copy()
        df["semantic_severity"] = severities
        df["semantic_score"] = confidences
        logger.info(
            f"Semantic scoring complete. Distribution:\n"
            f"{pd.Series(severities).value_counts().to_string()}"
        )
        return df
