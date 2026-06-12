"""
app.py
======
Support Integrity Auditor (SIA) — Streamlit Web Application
MARS Open Projects 2026

Pages:
  1. Home — Project overview, architecture, statistics
  2. Single Ticket Analysis — Real-time mismatch detection
  3. Batch CSV Analysis — Upload and analyze multiple tickets
  4. Analytics Dashboard — Distribution charts and heatmaps

Run:
  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800; color: #1a1a2e;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem; color: #555; text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; margin: 0.3rem;
    }
    .mismatch-badge {
        background: #ff4444; color: white; padding: 4px 12px;
        border-radius: 20px; font-weight: 700; font-size: 0.9rem;
    }
    .consistent-badge {
        background: #00c851; color: white; padding: 4px 12px;
        border-radius: 20px; font-weight: 700; font-size: 0.9rem;
    }
    .hidden-crisis {
        background: #ff6b35; color: white; padding: 6px 14px;
        border-radius: 8px; font-weight: 600;
    }
    .false-alarm {
        background: #ffa500; color: white; padding: 6px 14px;
        border-radius: 8px; font-weight: 600;
    }
    .evidence-item {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 0.6rem 1rem; margin: 0.3rem 0; border-radius: 4px;
        font-size: 0.9rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SIA model...")
def load_sia_model():
    """Loads the trained baseline classifier (or falls back to rule-based)."""
    try:
        from src.ml.baseline_model import BaselineClassifier
        model = BaselineClassifier.load()
        return model, "baseline"
    except Exception:
        return None, "rule_based"


@st.cache_data(show_spinner="Running analysis...")
def run_single_prediction(subject, description, priority, channel, ticket_type, resolution_time):
    """Cached single-ticket prediction."""
    from predict import predict_single_ticket
    return predict_single_ticket(
        subject, description, priority, channel, ticket_type, resolution_time
    )


@st.cache_data(show_spinner="Processing batch...")
def run_batch_prediction(csv_bytes: bytes):
    """Cached batch prediction on uploaded CSV."""
    import io
    from src.utils.data_loader import validate_and_clean
    from src.pseudo_labeling.fusion import PseudoLabelGenerator, create_mismatch_label
    from src.pseudo_labeling.signal_rule_based import RuleBasedSeverityScorer
    from src.pseudo_labeling.signal_resolution_time import ResolutionTimeSeverityScorer
    from src.dossier.generator import DossierGenerator

    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    df = validate_and_clean(df_raw)

    rule_scorer = RuleBasedSeverityScorer()
    rt_scorer = ResolutionTimeSeverityScorer()
    df = rule_scorer.score(df)
    rt_scorer.fit(df)
    df = rt_scorer.score(df)
    df["inferred_severity"] = df["rule_severity"]
    df["fusion_confidence"] = df["rule_score"]
    df["signal_agreement"] = 0.67
    df["semantic_severity"] = df["rule_severity"]
    df["semantic_score"] = df["rule_score"]

    from src.utils.config import COL_PRIORITY
    results = [
        create_mismatch_label(row[COL_PRIORITY], row["inferred_severity"])
        for _, row in df.iterrows()
    ]
    df["mismatch_label"] = [r[0] for r in results]
    df["mismatch_type"] = [r[1] for r in results]
    df["severity_delta"] = [r[2] for r in results]
    df["label"] = (df["mismatch_label"] == "Mismatch").astype(int)

    model, _ = load_sia_model()
    if model:
        try:
            probs = model.predict_proba(df)
            df["mismatch_probability"] = probs[:, 1]
            df["mismatch_label"] = np.where(probs[:, 1] >= 0.5, "Mismatch", "Consistent")
        except Exception:
            df["mismatch_probability"] = df["fusion_confidence"]

    dossier_gen = DossierGenerator()
    dossiers = dossier_gen.generate_batch(df)
    return df, dossiers


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## 🔍 SIA Navigator")
        st.markdown("---")
        page = st.radio(
            "Go to",
            ["🏠 Home", "🎫 Single Ticket", "📦 Batch Analysis", "📊 Dashboard"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**MARS Open Projects 2026**")
        st.markdown("Support Integrity Auditor")
        st.caption("v1.0 | DeBERTa + TF-IDF")
    return page


# ─────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────

def page_home():
    st.markdown('<div class="main-title">🔍 Support Integrity Auditor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Semantics-driven Priority Mismatch Detection for Enterprise CRM</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy Target", "≥ 83%", help="Binary classification accuracy")
    with col2:
        st.metric("Macro F1 Target", "≥ 0.82", help="Macro-averaged F1 score")
    with col3:
        st.metric("Per-Class Recall", "≥ 0.78", help="Both Consistent & Mismatch classes")
    with col4:
        st.metric("Signals Used", "3", help="Semantic + RT + Rule-Based")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("🎯 What is SIA?")
        st.markdown("""
        **SIA** automatically detects **Priority Mismatches** in customer support tickets — 
        cases where the human-assigned priority *conflicts* with the ticket's actual severity.

        **Two mismatch types:**
        - 🔴 **Hidden Crisis**: Critical issue mislabeled as Low/Medium — SLA risk!
        - 🟡 **False Alarm**: Trivial issue mislabeled as High/Critical — wastes resources!

        **Pipeline:**
        1. Self-supervised pseudo-label generation (3 signals)
        2. Fine-tuned DeBERTa-v3-small classifier
        3. Hallucination-free evidence dossiers
        """)

    with col_r:
        st.subheader("🏗️ Architecture")
        st.markdown("""
        ```
        Raw Tickets (CSV)
             │
             ▼
        ┌─────────────────────────────┐
        │   Pseudo-Label Generator    │
        │  Signal A: Semantic Embed   │
        │  Signal B: Resolution Time  │
        │  Signal C: Rule-Based NLP   │
        │  → Weighted Fusion          │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   Supervised Classifier     │
        │  v1: TF-IDF + LogReg        │
        │  v2: DeBERTa-v3-small       │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   Evidence Dossier Engine   │
        │  Zero-hallucination output  │
        └─────────────────────────────┘
        ```
        """)

    st.markdown("---")
    st.subheader("📋 Quick Examples")

    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        with st.container():
            st.markdown("**🔴 Hidden Crisis Example**")
            st.error(
                "**Ticket:** 'Our production payment gateway is down and customers cannot complete purchases.'\n\n"
                "**Assigned:** Low  →  **Actual:** Critical\n\n"
                "**Result:** MISMATCH — Hidden Crisis detected!"
            )
    with ex_col2:
        with st.container():
            st.markdown("**🟡 False Alarm Example**")
            st.warning(
                "**Ticket:** 'I want to change my profile picture.'\n\n"
                "**Assigned:** Critical  →  **Actual:** Low\n\n"
                "**Result:** MISMATCH — False Alarm detected!"
            )

    st.markdown("---")
    st.subheader("📊 Dataset")
    st.markdown("""
    | Column | Role |
    |--------|------|
    | Ticket Subject | Short summary of the issue |
    | Ticket Description | Full NL problem statement |
    | Ticket Priority | Human-assigned label (Low/Medium/High/Critical) |
    | Ticket Channel | Intake channel (email, chat, phone) |
    | Resolution Time | Time to resolve — indirect severity signal |
    | Ticket Type | Category of issue |
    """)


# ─────────────────────────────────────────────
# PAGE 2: SINGLE TICKET ANALYSIS
# ─────────────────────────────────────────────

def page_single():
    st.title("🎫 Single Ticket Analysis")
    st.markdown("Enter a support ticket to detect priority mismatches in real-time.")

    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Ticket Subject *", placeholder="e.g., Production server down")
            priority = st.selectbox("Assigned Priority *", ["Low", "Medium", "High", "Critical"])
            channel = st.selectbox("Channel", ["Email", "Chat", "Phone", "Social Media", "Portal"])
        with col2:
            ticket_type = st.selectbox("Ticket Type", [
                "Technical Issue", "Billing", "Feature Request",
                "Account Management", "Bug Report"
            ])
            resolution_time = st.number_input(
                "Resolution Time (hours)", min_value=0.1, max_value=720.0,
                value=24.0, step=0.5,
                help="Estimated or actual time to resolve."
            )
        description = st.text_area(
            "Ticket Description *",
            placeholder="Describe the issue in detail...",
            height=120,
        )
        submitted = st.form_submit_button("🔍 Analyze Ticket", use_container_width=True)

    if submitted:
        if not subject.strip() or not description.strip():
            st.error("Please fill in Subject and Description.")
            return

        with st.spinner("Analyzing ticket..."):
            result = run_single_prediction(
                subject, description, priority, channel, ticket_type, resolution_time
            )

        # ── Result Header ──
        st.markdown("---")
        mismatch = result["mismatch_label"] == "Mismatch"
        mismatch_type = result.get("mismatch_type", "")
        inferred = result["inferred_severity"]
        confidence = result["prediction"]["confidence"]

        if mismatch:
            if mismatch_type == "Hidden Crisis":
                st.error(f"🔴 **MISMATCH DETECTED — Hidden Crisis**")
                st.markdown(f"Ticket assigned **{priority}** but actual severity is **{inferred}**")
            else:
                st.warning(f"🟡 **MISMATCH DETECTED — False Alarm**")
                st.markdown(f"Ticket assigned **{priority}** but actual severity is only **{inferred}**")
        else:
            st.success(f"✅ **CONSISTENT** — Assigned priority matches inferred severity.")

        # ── Metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Assigned Priority", priority)
        m2.metric("Inferred Severity", inferred)
        m3.metric("Severity Delta", str(result["severity_delta"]))
        m4.metric("Confidence", f"{confidence:.1%}")

        # ── Dossier ──
        if result.get("dossier"):
            st.markdown("---")
            st.subheader("📋 Evidence Dossier")
            dossier = result["dossier"]

            doss_col1, doss_col2 = st.columns([2, 1])
            with doss_col1:
                st.markdown("**Feature Evidence:**")
                for ev in dossier.get("feature_evidence", []):
                    signal = ev.get("signal", "")
                    if signal == "keyword":
                        st.markdown(
                            f'<div class="evidence-item">🔑 <b>Keyword</b>: "{ev["value"]}" '
                            f'— {ev["weight"]} <i>(from {ev.get("source_field","")})</i></div>',
                            unsafe_allow_html=True
                        )
                    elif signal == "resolution_time":
                        st.markdown(
                            f'<div class="evidence-item">⏱ <b>Resolution Time</b>: {ev["value"]} '
                            f'— {ev["interpretation"]}</div>',
                            unsafe_allow_html=True
                        )
                    elif signal == "semantic_embedding":
                        st.markdown(
                            f'<div class="evidence-item">🧠 <b>Semantic</b>: {ev["value"]}</div>',
                            unsafe_allow_html=True
                        )

                st.markdown("**Constraint Analysis:**")
                st.info(dossier.get("constraint_analysis", ""))

            with doss_col2:
                st.markdown("**Dossier Summary:**")
                st.json({
                    "ticket_id": dossier.get("ticket_id"),
                    "mismatch_type": dossier.get("mismatch_type"),
                    "severity_delta": dossier.get("severity_delta"),
                    "confidence": dossier.get("confidence"),
                })

            # Download dossier
            dossier_json = json.dumps(dossier, indent=2)
            st.download_button(
                "⬇️ Download Dossier JSON",
                data=dossier_json,
                file_name=f"dossier_{dossier.get('ticket_id','ticket')}.json",
                mime="application/json",
            )


# ─────────────────────────────────────────────
# PAGE 3: BATCH CSV ANALYSIS
# ─────────────────────────────────────────────

def page_batch():
    st.title("📦 Batch CSV Analysis")
    st.markdown("Upload a CSV with multiple tickets for bulk mismatch detection.")

    with st.expander("📋 Required CSV Format", expanded=False):
        st.markdown("""
        Your CSV must contain these columns (exact names):

        | Column | Required | Description |
        |--------|----------|-------------|
        | Ticket Subject | ✅ | Short issue summary |
        | Ticket Description | ✅ | Full description |
        | Ticket Priority | ✅ | Low / Medium / High / Critical |
        | Ticket Channel | Optional | Email, Chat, Phone, etc. |
        | Resolution Time | Optional | Hours (numeric) |
        | Ticket Type | Optional | Technical Issue, etc. |
        """)

        # Sample download
        sample = pd.DataFrame([
            {"Ticket Subject": "Server is down", "Ticket Description": "Production outage affecting all users", "Ticket Priority": "Low", "Ticket Channel": "Email", "Resolution Time": 2.0, "Ticket Type": "Technical Issue"},
            {"Ticket Subject": "Change profile picture", "Ticket Description": "I want to update my avatar", "Ticket Priority": "Critical", "Ticket Channel": "Chat", "Resolution Time": 96.0, "Ticket Type": "Account Management"},
        ])
        st.download_button(
            "⬇️ Download Sample CSV",
            data=sample.to_csv(index=False),
            file_name="sample_tickets.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload Tickets CSV", type=["csv"])

    if uploaded:
        csv_bytes = uploaded.read()
        df_result, dossiers = run_batch_prediction(csv_bytes)

        mismatch_count = (df_result["mismatch_label"] == "Mismatch").sum()
        total = len(df_result)
        hidden_crisis = (df_result.get("mismatch_type", pd.Series()) == "Hidden Crisis").sum()
        false_alarm = (df_result.get("mismatch_type", pd.Series()) == "False Alarm").sum()

        # ── Summary Metrics ──
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets", total)
        m2.metric("Mismatches", f"{mismatch_count} ({mismatch_count/total*100:.0f}%)")
        m3.metric("Hidden Crisis 🔴", hidden_crisis)
        m4.metric("False Alarm 🟡", false_alarm)

        # ── Results Table ──
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        display_cols = [
            "Ticket ID", "Ticket Subject", "Ticket Priority",
            "inferred_severity", "mismatch_label", "mismatch_type", "severity_delta",
        ]
        display_cols = [c for c in display_cols if c in df_result.columns]
        display_df = df_result[display_cols].copy()

        def color_mismatch(val):
            if val == "Mismatch":
                return "background-color: #ffe0e0"
            elif val == "Consistent":
                return "background-color: #e0ffe0"
            return ""

        if "mismatch_label" in display_df.columns:
            styled = display_df.style.applymap(color_mismatch, subset=["mismatch_label"])
            st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.dataframe(display_df, use_container_width=True, height=400)

        # ── Downloads ──
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=df_result[display_cols].to_csv(index=False),
                file_name="sia_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇️ Download All Dossiers JSON",
                data=json.dumps(dossiers, indent=2),
                file_name="sia_dossiers.json",
                mime="application/json",
                use_container_width=True,
            )

        # ── Charts ──
        st.markdown("---")
        st.subheader("📈 Quick Analytics")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if "mismatch_label" in df_result.columns:
                fig = px.pie(
                    df_result["mismatch_label"].value_counts().reset_index(),
                    names="mismatch_label", values="count",
                    title="Mismatch Distribution",
                    color_discrete_map={"Mismatch": "#ff4444", "Consistent": "#00c851"},
                )
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            if "inferred_severity" in df_result.columns:
                sev_counts = df_result["inferred_severity"].value_counts().reset_index()
                fig2 = px.bar(
                    sev_counts, x="inferred_severity", y="count",
                    title="Inferred Severity Distribution",
                    color="inferred_severity",
                    color_discrete_map={
                        "Critical": "#dc3545", "High": "#fd7e14",
                        "Medium": "#ffc107", "Low": "#28a745",
                    },
                )
                st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 4: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────

def page_dashboard():
    st.title("📊 Analytics Dashboard")
    st.markdown("Priority Mismatch intelligence across your support ticket dataset.")

    # Load sample data for demo
    @st.cache_data
    def get_demo_data():
        from src.utils.data_loader import generate_synthetic_dataset
        from src.utils.data_loader import validate_and_clean
        from src.pseudo_labeling.signal_rule_based import RuleBasedSeverityScorer
        from src.pseudo_labeling.signal_resolution_time import ResolutionTimeSeverityScorer
        from src.pseudo_labeling.fusion import create_mismatch_label
        from src.utils.config import COL_PRIORITY

        df = generate_synthetic_dataset(n_tickets=500)
        df = validate_and_clean(df)
        rule = RuleBasedSeverityScorer()
        df = rule.score(df)
        rt = ResolutionTimeSeverityScorer()
        rt.fit(df)
        df = rt.score(df)
        df["inferred_severity"] = df["rule_severity"]

        results = [create_mismatch_label(row[COL_PRIORITY], row["inferred_severity"])
                   for _, row in df.iterrows()]
        df["mismatch_label"] = [r[0] for r in results]
        df["mismatch_type"] = [r[1] for r in results]
        df["severity_delta"] = [r[2] for r in results]
        return df

    with st.spinner("Loading dashboard data..."):
        df = get_demo_data()

    total = len(df)
    mismatches = (df["mismatch_label"] == "Mismatch").sum()
    hidden_crisis = (df["mismatch_type"] == "Hidden Crisis").sum()
    false_alarm = (df["mismatch_type"] == "False Alarm").sum()

    # ── Top KPIs ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Tickets", f"{total:,}")
    k2.metric("Mismatches", f"{mismatches:,}", delta=f"{mismatches/total*100:.0f}%")
    k3.metric("🔴 Hidden Crisis", hidden_crisis)
    k4.metric("🟡 False Alarm", false_alarm)
    k5.metric("Consistent", total - mismatches)

    st.markdown("---")

    # ── Row 1: Priority Distribution + Mismatch by Priority ──
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        priority_counts = df["Ticket Priority"].value_counts().reset_index()
        fig = px.bar(
            priority_counts, x="Ticket Priority", y="count",
            title="Assigned Priority Distribution",
            color="Ticket Priority",
            color_discrete_map={
                "Critical": "#dc3545", "High": "#fd7e14",
                "Medium": "#ffc107", "Low": "#28a745",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        mismatch_by_priority = df.groupby("Ticket Priority")["mismatch_label"].apply(
            lambda x: (x == "Mismatch").mean() * 100
        ).reset_index()
        mismatch_by_priority.columns = ["Priority", "Mismatch Rate (%)"]
        fig2 = px.bar(
            mismatch_by_priority, x="Priority", y="Mismatch Rate (%)",
            title="Mismatch Rate by Assigned Priority",
            color="Priority",
            color_discrete_map={
                "Critical": "#dc3545", "High": "#fd7e14",
                "Medium": "#ffc107", "Low": "#28a745",
            },
        )
        fig2.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% threshold")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Mismatch Type + Inferred Severity Distribution ──
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        type_counts = df["mismatch_type"].replace("", "Consistent").value_counts().reset_index()
        fig3 = px.pie(
            type_counts, names="mismatch_type", values="count",
            title="Mismatch Type Distribution",
            color_discrete_map={
                "Hidden Crisis": "#ff4444",
                "False Alarm": "#ffa500",
                "Consistent": "#00c851",
            },
        )
        st.plotly_chart(fig3, use_container_width=True)

    with row2_col2:
        sev_dist = df["inferred_severity"].value_counts().reset_index()
        fig4 = px.bar(
            sev_dist, x="inferred_severity", y="count",
            title="Inferred Severity Distribution",
            color="inferred_severity",
            color_discrete_map={
                "Critical": "#dc3545", "High": "#fd7e14",
                "Medium": "#ffc107", "Low": "#28a745",
            },
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Channel Analysis + Severity Delta Heatmap ──
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        if "Ticket Channel" in df.columns:
            channel_mismatch = df.groupby("Ticket Channel")["mismatch_label"].apply(
                lambda x: (x == "Mismatch").sum()
            ).reset_index()
            channel_mismatch.columns = ["Channel", "Mismatches"]
            fig5 = px.bar(
                channel_mismatch, x="Channel", y="Mismatches",
                title="Mismatches by Channel",
                color="Mismatches", color_continuous_scale="Reds",
            )
            st.plotly_chart(fig5, use_container_width=True)

    with row3_col2:
        # Severity delta heatmap
        if "Ticket Channel" in df.columns and "inferred_severity" in df.columns:
            from src.utils.config import PRIORITY_MAP
            df["inferred_rank"] = df["inferred_severity"].map(PRIORITY_MAP).fillna(1)
            df["assigned_rank"] = df["Ticket Priority"].map(PRIORITY_MAP).fillna(1)
            pivot = df.pivot_table(
                values="severity_delta",
                index="Ticket Priority",
                columns="inferred_severity",
                aggfunc="count",
                fill_value=0,
            )
            fig6 = px.imshow(
                pivot,
                title="Severity Delta Heatmap\n(Assigned Priority vs Inferred Severity)",
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            st.plotly_chart(fig6, use_container_width=True)

    # ── Row 4: Ticket Type Analysis ──
    st.markdown("---")
    if "Ticket Type" in df.columns:
        type_mismatch = df.groupby("Ticket Type").agg(
            Total=("mismatch_label", "count"),
            Mismatches=("mismatch_label", lambda x: (x == "Mismatch").sum()),
        ).reset_index()
        type_mismatch["Mismatch Rate"] = (type_mismatch["Mismatches"] / type_mismatch["Total"] * 100).round(1)

        fig7 = px.scatter(
            type_mismatch, x="Total", y="Mismatch Rate",
            size="Mismatches", color="Ticket Type",
            title="Ticket Type: Volume vs Mismatch Rate",
            text="Ticket Type",
            labels={"Total": "Total Tickets", "Mismatch Rate": "Mismatch Rate (%)"},
        )
        st.plotly_chart(fig7, use_container_width=True)

    # ── Resolution Time Trend ──
    st.subheader("⏱ Resolution Time Distribution by Inferred Severity")
    if "Resolution Time" in df.columns:
        fig8 = px.box(
            df[df["Resolution Time"] <= 200],
            x="inferred_severity", y="Resolution Time",
            color="inferred_severity",
            title="Resolution Time (hours) by Inferred Severity",
            category_orders={"inferred_severity": ["Low", "Medium", "High", "Critical"]},
            color_discrete_map={
                "Critical": "#dc3545", "High": "#fd7e14",
                "Medium": "#ffc107", "Low": "#28a745",
            },
        )
        st.plotly_chart(fig8, use_container_width=True)


# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────

def main():
    page = sidebar()

    if "Home" in page:
        page_home()
    elif "Single" in page:
        page_single()
    elif "Batch" in page:
        page_batch()
    elif "Dashboard" in page:
        page_dashboard()


if __name__ == "__main__":
    main()
