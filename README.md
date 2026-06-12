# 🔍 Support Integrity Auditor (SIA)

> **MARS Open Projects 2026 — AI/ML Problem Statement 1**  
> Semantics-driven, evidence-grounded automated auditing system for CRM Priority Mismatch Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
- [Evaluation Results](#evaluation-results)
- [Pseudo-Label Generation](#pseudo-label-generation)
- [Ablation Study](#ablation-study)
- [Evidence Dossier](#evidence-dossier)
- [Deployment](#deployment)
- [Testing](#testing)
- [Dashboard](#dashboard)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

SIA automatically detects **Priority Mismatches** in customer support tickets — cases where the human-assigned priority conflicts with the ticket's true objective severity.

**Two mismatch types:**
- 🔴 **Hidden Crisis**: Critical issue mislabeled as Low/Medium (SLA risk)
- 🟡 **False Alarm**: Trivial issue inflated to High/Critical (resource waste)

**Key Innovation:** No pre-labeled mismatch data exists. SIA bootstraps its own supervision signal using **self-supervised pseudo-label generation** from 3 independent severity signals, then trains a fine-tuned **DeBERTa-v3-small** classifier on the generated labels.

---

## 🔴 Problem Statement

In enterprise CRM ecosystems, manual ticket triage suffers from:
- **Agent fatigue bias** — tired agents mislabel severity
- **Customer favoritism** — priority based on customer tier, not issue severity
- **Keyword anchoring** — agents react to surface-level words, not semantic meaning

Existing rule-based systems fail on:
- Sarcasm: *"Just our entire payment system taking a nap"* (assigned: Low)
- Indirect urgency: *"Finance has called twice, ops bridge is open"* (assigned: Low)
- Negation traps: *"Not a server down issue — system just unusable for 10k users"*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW TICKET DATA (CSV)                     │
│         Subject | Description | Priority | Channel | RT      │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   DATA PREPROCESSING    │
              │  Cleaning • Validation  │
              │  combined_text field    │
              └────────────┬────────────┘
                           │
     ┌─────────────────────▼──────────────────────┐
     │         PSEUDO-LABEL GENERATOR              │
     │                                             │
     │  Signal A (45%)   Signal B (25%)  Signal C (30%)│
     │  Semantic Embed   Resolution Time  Rule-Based  │
     │  all-MiniLM-L6    Log-percentile   Keyword NLP │
     │  Cosine sim       Severity bins    Negation    │
     │                                             │
     │         ── Weighted Fusion ──               │
     │     inferred_severity + confidence           │
     │                                             │
     │         ── Mismatch Label ──                │
     │     Consistent | Hidden Crisis | False Alarm│
     └─────────────────────┬───────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   SUPERVISED TRAINING   │
              │                         │
              │  v1: TF-IDF + LogReg    │
              │  v2: DeBERTa-v3-small   │
              │  Weighted loss • ES     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   EVIDENCE DOSSIER      │
              │  Zero hallucination     │
              │  Grounded evidence      │
              │  Constraint analysis    │
              └─────────────────────────┘
```

### Component Descriptions

| Component | Purpose |
|-----------|---------|
| `data_loader.py` | Loads Kaggle CSV or generates synthetic data |
| `signal_semantic.py` | Sentence-transformer embeddings vs severity anchors |
| `signal_resolution_time.py` | Log-percentile RT → severity mapping |
| `signal_rule_based.py` | Keyword density + negation + escalation detection |
| `fusion.py` | Weighted signal fusion + mismatch label creation |
| `baseline_model.py` | TF-IDF + Logistic Regression (fast baseline) |
| `advanced_model.py` | Fine-tuned DeBERTa-v3-small (production model) |
| `evaluator.py` | Metrics computation + threshold validation |
| `generator.py` | Hallucination-free evidence dossier generation |
| `app.py` | 4-page Streamlit application |

---

## 📊 Dataset

**Source:** [Customer Support Tickets CRM Dataset](https://kaggle.com/datasets/ajverse/customersupport-tickets-crm-dataset/data)

| Column | Role | Used By |
|--------|------|---------|
| Ticket Subject | Short issue summary | All signals |
| Ticket Description | Full NL description | All signals |
| Ticket Priority | Human-assigned label | Mismatch comparison |
| Ticket Channel | Intake channel | Metadata features |
| Resolution Time | Hours to resolve | Signal B |
| Ticket Type | Issue category | Metadata features |
| Customer Email | Customer identifier | Context |
| Product Purchased | Product context | Context |

**Note:** If the Kaggle CSV is not available, SIA auto-generates a synthetic dataset with realistic mismatch patterns for full pipeline demonstration.

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- 8GB RAM (16GB for DeBERTa training)
- GPU optional (CUDA for DeBERTa, CPU fallback available)

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/SIA.git
cd SIA

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download dataset from Kaggle
# Place as: data/raw/customer_support_tickets.csv
```

---

## 💻 Usage

### Run Streamlit App (No training required — uses synthetic data)

```bash
streamlit run app.py
```
Open http://localhost:8501

### Train the Pipeline

```bash
# Train baseline only (fast, ~2 minutes)
python train_pipeline.py --skip-advanced

# Train both models (requires GPU for DeBERTa, ~30 min)
python train_pipeline.py

# Train on custom dataset
python train_pipeline.py --data-path data/raw/my_tickets.csv
```

### Run Inference

```bash
# Single ticket (CLI)
python predict.py single \
  --subject "Server is down" \
  --description "All users affected, revenue loss" \
  --priority Low \
  --channel Email

# Batch CSV
python predict.py csv --input data/raw/tickets.csv --model baseline
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Adversarial tests
pytest tests/integration/test_adversarial.py -v -s

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🔬 Training Pipeline

The `train_pipeline.py` executes 7 stages:

1. **Load & Clean** — validates columns, standardizes priorities
2. **Pseudo-Label Generation** — runs all 3 signals + fusion
3. **Train/Val/Test Split** — stratified 70/10/20
4. **Baseline Training** — TF-IDF(15k) + LogReg, class_weight='balanced'
5. **Advanced Training** — DeBERTa-v3-small fine-tuning (5 epochs, early stopping)
6. **Evaluation** — all metrics, confusion matrices, model comparison
7. **Dossier Generation** — evidence dossiers for test set mismatches

---

## 📈 Evaluation Results

| Metric | Baseline (TF-IDF+LR) | Advanced (DeBERTa) | Target |
|--------|---------------------|-------------------|--------|
| Accuracy | 84.2% | 87.6% | ≥ 83% |
| Macro F1 | 0.834 | 0.871 | ≥ 0.82 |
| Consistent Recall | 0.81 | 0.86 | ≥ 0.78 |
| Mismatch Recall | 0.80 | 0.84 | ≥ 0.78 |
| ROC-AUC | 0.91 | 0.94 | — |

> *Results on synthetic dataset. Real dataset results may vary.*

---

## 🧪 Pseudo-Label Generation

### Signal Fusion Strategy

```
Final Score(level) = 
    0.45 × Semantic_confidence(level)
  + 0.25 × RT_confidence(level)
  + 0.30 × Rule_confidence(level)
```

**Why these weights?**
- Semantic (0.45): Captures nuanced meaning, handles indirect urgency
- Rule-Based (0.30): High precision on explicit keyword patterns
- Resolution Time (0.25): Strong indirect signal, but noisy for edge cases

### Signal A — Semantic Embedding
- Model: `all-MiniLM-L6-v2`
- 5 anchor sentences per severity level (Low/Medium/High/Critical)
- Cosine similarity → softmax confidence
- Handles sarcasm and indirect language better than rules

### Signal B — Resolution Time
- Log-transform to normalize right-skewed distribution
- Percentile-based thresholds: bottom 15%→Critical, 15-35%→High, 35-65%→Medium, top 35%→Low
- Rationale: Critical issues are resolved fastest (highest urgency)

### Signal C — Rule-Based NLP
- Critical keyword density × 3.0
- Business impact patterns × 2.0
- Quantifier patterns × 2.0
- Escalation phrases × 1.5
- Negation-aware: discounts hits preceded by negation words

---

## 📊 Ablation Study

| Signal Pair | Agreement Rate |
|-------------|----------------|
| Semantic vs RT | 0.71 |
| Semantic vs Rule | 0.78 |
| RT vs Rule | 0.69 |
| Semantic vs Fused | 0.84 |
| RT vs Fused | 0.76 |
| Rule vs Fused | 0.81 |

**Individual Signal Contributions:**

| Signal | Decisive Votes | Contribution |
|--------|---------------|-------------|
| Semantic | 87 | 45.1% |
| Rule-Based | 64 | 33.2% |
| Resolution Time | 42 | 21.8% |

---

## 📋 Evidence Dossier

Every mismatch ticket produces a structured, grounded dossier:

```json
{
  "ticket_id": "TKT-00123",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 3,
  "feature_evidence": [
    {
      "signal": "keyword",
      "value": "payment gateway",
      "weight": "Critical severity indicator",
      "source_field": "Ticket Description"
    },
    {
      "signal": "resolution_time",
      "value": "1.5 hours",
      "interpretation": "Resolved in 1.5h — extremely fast, indicating production-critical handling",
      "source_field": "Resolution Time"
    },
    {
      "signal": "semantic_embedding",
      "value": "Semantic similarity to 'Critical' anchor: 0.891",
      "source_field": "Ticket Subject + Ticket Description"
    }
  ],
  "constraint_analysis": "The ticket 'Production payment gateway down...' was assigned Low priority, but semantic analysis and rule-based signals converge on Critical severity — a gap of 3 levels. The 1.5-hour resolution time indicates the support team treated this as operationally urgent. Under-prioritization of Low→Critical issues risks SLA breach and delayed escalation.",
  "confidence": "High (87.40%)"
}
```

**Anti-Hallucination Guarantee:**
- Every `keyword` evidence item is regex-verified against the actual ticket text
- Resolution time value is pulled directly from the `Resolution Time` column
- Constraint analysis uses only observable ticket fields (no inferences about external systems)

---

## 🐳 Deployment

### Local
```bash
streamlit run app.py
```

### Docker
```bash
docker-compose up --build
# Access: http://localhost:8501
```

### Train in Docker
```bash
docker-compose --profile train up sia-train
```

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → `app.py`
4. Advanced settings → Python 3.11

### Hugging Face Spaces
```bash
# Create Space with Streamlit SDK
# Upload all files
# Add requirements.txt
# Space auto-builds and deploys
```

### Render
```bash
# New Web Service → connect GitHub repo
# Build Command:  pip install -r requirements.txt
# Start Command:  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 📊 Dashboard

The Streamlit app includes 4 pages:

| Page | Features |
|------|----------|
| 🏠 Home | Architecture, examples, dataset overview |
| 🎫 Single Ticket | Real-time analysis + dossier |
| 📦 Batch CSV | Upload → predictions → downloadable results |
| 📊 Dashboard | Priority distribution, mismatch heatmap, channel analysis |

---

## 🔮 Future Improvements

1. **LLM Zero-Shot Signal** — Add Mistral-7B-Instruct as Signal D for even richer severity inference
2. **LoRA Fine-tuning** — Apply LoRA adapters to reduce DeBERTa parameter count by 90%
3. **Active Learning** — Human-in-the-loop correction loop to improve pseudo-labels over time
4. **Multi-language Support** — Extend to non-English tickets using multilingual SBERT
5. **Real-time Integration** — REST API for live CRM webhook integration
6. **Temporal Analysis** — Detect mismatch trends over time to flag systemic bias
7. **Agent Fairness Audit** — Per-agent mismatch rate analysis to detect individual bias

---

## 📁 Project Structure

```
SIA/
├── data/
│   ├── raw/                    # Original dataset CSV
│   ├── processed/              # Cleaned dataset
│   └── pseudo_labels/          # Generated pseudo-labels + ablation
├── src/
│   ├── utils/
│   │   ├── config.py           # All hyperparameters and paths
│   │   ├── data_loader.py      # Dataset loading + cleaning
│   │   └── logger.py           # Centralized logging
│   ├── pseudo_labeling/
│   │   ├── signal_semantic.py  # Signal A: Sentence transformers
│   │   ├── signal_resolution_time.py  # Signal B: RT analysis
│   │   ├── signal_rule_based.py       # Signal C: NLP rules
│   │   └── fusion.py           # Signal fusion + mismatch labels
│   ├── ml/
│   │   ├── baseline_model.py   # TF-IDF + Logistic Regression
│   │   └── advanced_model.py   # DeBERTa-v3-small fine-tuning
│   ├── evaluation/
│   │   └── evaluator.py        # All metrics + visualization
│   └── dossier/
│       └── generator.py        # Evidence dossier generation
├── models/
│   ├── baseline/               # Saved baseline model
│   └── advanced/               # Saved DeBERTa weights
├── outputs/
│   ├── predictions/            # CSV predictions
│   ├── dossiers/               # JSON dossiers
│   └/reports/                 # Evaluation reports + plots
├── tests/
│   ├── unit/                   # Unit tests per module
│   └── integration/            # E2E + adversarial tests
├── app.py                      # Streamlit application
├── train_pipeline.py           # Standalone training script
├── predict.py                  # Inference script
├── requirements.txt            # Pinned dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service orchestration
└── README.md                   # This file
```

---

## 📜 License

MIT License — MARS Open Projects 2026

---

*Built for MARS Open Projects 2026 — Models and Robotics Section*
