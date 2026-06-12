"""
advanced_model.py
=================
VERSION 2: Fine-tuned DeBERTa-v3-small Classifier.

Architecture:
  - Base: microsoft/deberta-v3-small (pre-trained)
  - Input: [CLS] subject [SEP] description [SEP] + metadata tokens
  - Head: Classification head → binary output (Consistent / Mismatch)
  - Metadata: Channel, ticket_type appended as prefix tokens
  - Training: Weighted cross-entropy loss for class imbalance
  - Regularization: Early stopping + model checkpointing

This file handles:
  - Dataset preparation and tokenization
  - Model definition (DeBERTaForMismatchClassification)
  - Training loop with early stopping
  - Evaluation during training
  - Model save/load
  - Single ticket inference
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.utils.config import (
    ADVANCED_MODEL_DIR, DEBERTA_MODEL_NAME, DEBERTA_MAX_LENGTH,
    DEBERTA_BATCH_SIZE, DEBERTA_LEARNING_RATE, DEBERTA_NUM_EPOCHS,
    DEBERTA_WARMUP_RATIO, DEBERTA_WEIGHT_DECAY, DEBERTA_EARLY_STOPPING_PATIENCE,
    RANDOM_SEED, COL_CHANNEL, COL_TICKET_TYPE, COL_RESOLUTION_TIME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────

class TicketDataset(Dataset):
    """
    PyTorch Dataset for support ticket classification.

    Input format:
      "[CHANNEL: {channel}] [TYPE: {ticket_type}] {subject} {description}"

    Metadata is prepended as special context tokens so the model
    can attend to structured fields alongside the text.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        labels: Optional[pd.Series] = None,
        max_length: int = DEBERTA_MAX_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels.values if labels is not None else None

        # Build input strings with metadata prefix
        self.texts = [
            self._build_input(row)
            for _, row in df.iterrows()
        ]

    def _build_input(self, row: pd.Series) -> str:
        """Builds the model input string for one ticket."""
        channel = str(row.get(COL_CHANNEL, "Unknown"))
        ticket_type = str(row.get(COL_TICKET_TYPE, "Unknown"))
        rt = float(row.get(COL_RESOLUTION_TIME, 24.0))
        subject = str(row.get("Ticket Subject", "")).strip()
        description = str(row.get("Ticket Description", "")).strip()

        # RT bucket as interpretable token
        if rt <= 4:
            rt_token = "RESOLVED-FAST"
        elif rt <= 24:
            rt_token = "RESOLVED-NORMAL"
        elif rt <= 72:
            rt_token = "RESOLVED-SLOW"
        else:
            rt_token = "RESOLVED-VERY-SLOW"

        return (
            f"[CHANNEL: {channel}] [TYPE: {ticket_type}] [RT: {rt_token}] "
            f"{subject} {self.tokenizer.sep_token} {description}"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────

class DeBERTaMismatchClassifier(nn.Module):
    """
    DeBERTa-v3-small with classification head for mismatch detection.

    Architecture:
      DeBERTa Encoder → [CLS] pooled → Dropout → Linear(768, 256)
      → GELU → Dropout → Linear(256, 2) → Logits
    """

    def __init__(self, model_name: str = DEBERTA_MODEL_NAME, num_labels: int = 2):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Optional token type IDs.
            labels: Optional ground-truth labels for loss computation.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result


# ─────────────────────────────────────────────
# TRAINER CLASS
# ─────────────────────────────────────────────

class AdvancedModelTrainer:
    """
    Manages DeBERTa fine-tuning with early stopping and checkpointing.

    Usage:
        trainer = AdvancedModelTrainer()
        trainer.train(train_df, train_labels, val_df, val_labels)
        preds = trainer.predict(test_df)
        trainer.save()
    """

    def __init__(self, model_name: str = DEBERTA_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._best_val_f1 = 0.0
        self._no_improve_count = 0

        logger.info(f"Using device: {self.device}")

    def _load_tokenizer(self):
        """Loads tokenizer (lazy)."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _compute_class_weights(self, labels: pd.Series) -> torch.Tensor:
        """Computes inverse-frequency class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=labels.values,
        )
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train(
        self,
        train_df: pd.DataFrame,
        train_labels: pd.Series,
        val_df: pd.DataFrame,
        val_labels: pd.Series,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, list]:
        """
        Fine-tunes DeBERTa on pseudo-labeled data.

        Args:
            train_df: Training DataFrame.
            train_labels: Binary training labels.
            val_df: Validation DataFrame.
            val_labels: Binary validation labels.
            class_weights: Optional pre-computed class weights.

        Returns:
            Training history dict with loss and F1 per epoch.
        """
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        from sklearn.metrics import f1_score

        self._load_tokenizer()

        # Datasets
        train_dataset = TicketDataset(train_df, self.tokenizer, train_labels)
        val_dataset = TicketDataset(val_df, self.tokenizer, val_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=DEBERTA_BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=DEBERTA_BATCH_SIZE * 2, shuffle=False, num_workers=0,
        )

        # Model
        self.model = DeBERTaMismatchClassifier(self.model_name).to(self.device)

        # Class-weighted loss
        cw = class_weights if class_weights is not None else self._compute_class_weights(train_labels)
        loss_fn = nn.CrossEntropyLoss(weight=cw)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=DEBERTA_LEARNING_RATE,
            weight_decay=DEBERTA_WEIGHT_DECAY,
        )
        total_steps = len(train_loader) * DEBERTA_NUM_EPOCHS
        warmup_steps = int(total_steps * DEBERTA_WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_model_path = ADVANCED_MODEL_DIR / "best_checkpoint.pt"
        ADVANCED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, DEBERTA_NUM_EPOCHS + 1):
            # ── Training ──
            self.model.train()
            train_losses = []
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(outputs["logits"], labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # ── Validation ──
            self.model.eval()
            val_losses, val_preds, val_true = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self.device)

                    outputs = self.model(input_ids, attention_mask, token_type_ids)
                    vloss = loss_fn(outputs["logits"], labels)
                    val_losses.append(vloss.item())
                    preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(labels.cpu().numpy())

            avg_val_loss = np.mean(val_losses)
            val_f1 = f1_score(val_true, val_preds, average="macro")

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_f1"].append(val_f1)

            logger.info(
                f"Epoch {epoch}/{DEBERTA_NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Macro F1: {val_f1:.4f}"
            )

            # ── Early Stopping ──
            if val_f1 > self._best_val_f1:
                self._best_val_f1 = val_f1
                self._no_improve_count = 0
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"  ✓ New best model saved (Val F1: {val_f1:.4f})")
            else:
                self._no_improve_count += 1
                logger.info(f"  No improvement ({self._no_improve_count}/{DEBERTA_EARLY_STOPPING_PATIENCE})")
                if self._no_improve_count >= DEBERTA_EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        # Load best checkpoint
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        logger.info(f"Training complete. Best Val F1: {self._best_val_f1:.4f}")
        return history

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (predictions, probabilities).
        """
        self._load_tokenizer()
        dataset = TicketDataset(df, self.tokenizer)
        loader = DataLoader(dataset, batch_size=DEBERTA_BATCH_SIZE * 2, shuffle=False)

        self.model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                outputs = self.model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                all_preds.extend(preds)
                all_probs.extend(probs)

        return np.array(all_preds), np.array(all_probs)

    def predict_single(self, subject: str, description: str,
                        channel: str = "Email", ticket_type: str = "Technical Issue",
                        resolution_time: float = 24.0) -> Dict:
        """Single-ticket inference for the Streamlit app."""
        row = {
            "Ticket Subject": subject,
            "Ticket Description": description,
            COL_CHANNEL: channel,
            COL_TICKET_TYPE: ticket_type,
            COL_RESOLUTION_TIME: resolution_time,
            "combined_text": f"{subject.lower()} {description.lower()}",
        }
        single_df = pd.DataFrame([row])
        preds, probs = self.predict(single_df)
        pred = int(preds[0])
        prob_mismatch = float(probs[0][1])
        return {
            "prediction": pred,
            "label": "Mismatch" if pred == 1 else "Consistent",
            "confidence": round(prob_mismatch if pred == 1 else 1 - prob_mismatch, 4),
            "mismatch_probability": round(prob_mismatch, 4),
        }

    def save(self, path: Path = None):
        """Saves model and tokenizer."""
        save_dir = path or ADVANCED_MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model_weights.pt")
        self.tokenizer.save_pretrained(str(save_dir))
        config = {"model_name": self.model_name, "best_val_f1": self._best_val_f1}
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Advanced model saved to: {save_dir}")

    @classmethod
    def load(cls, path: Path = None) -> "AdvancedModelTrainer":
        """Loads a saved DeBERTa model."""
        from transformers import AutoTokenizer
        load_dir = path or ADVANCED_MODEL_DIR
        config_path = load_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        trainer = cls(config["model_name"])
        trainer._load_tokenizer()
        trainer.model = DeBERTaMismatchClassifier(config["model_name"]).to(trainer.device)
        trainer.model.load_state_dict(
            torch.load(load_dir / "model_weights.pt", map_location=trainer.device)
        )
        trainer.model.eval()
        trainer._best_val_f1 = config.get("best_val_f1", 0.0)
        logger.info(f"Advanced model loaded from: {load_dir}")
        return trainer
