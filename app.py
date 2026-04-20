"""
Streamlit UI — Twitter Customer Support Classifier
Talks to the FastAPI backend at localhost:8000
"""

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

COMPANY_EMOJI = {
    "AmazonHelp": "📦",
    "AppleSupport": "🍎",
    "Uber_Support": "🚗",
    "SpotifyCares": "🎵",
    "Delta": "✈️",
    "AmericanAir": "✈️",
    "TMobileHelp": "📱",
    "comcastcares": "📺",
    "SouthwestAir": "✈️",
    "VirginTrains": "🚂",
}

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎯",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Twitter Customer Support Classifier")
st.markdown("Classifies a customer tweet to the most likely support team using a **LinearSVC + TF-IDF** model.")

# ── Sidebar — API health & model info ─────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        requests.get(f"{API_URL}/health", timeout=3).json()
        info = requests.get(f"{API_URL}/model/info", timeout=3).json()
        st.success("API is online")
        st.metric("Model", info["model_type"])
        st.metric("Features", f"{info['num_features']:,}")
        st.metric("Classes", len(info["classes"]))
        st.markdown("**Supported companies:**")
        for c in sorted(info["classes"]):
            st.markdown(f"- {COMPANY_EMOJI.get(c, '🏢')} {c}")
    except Exception:
        st.error("API offline — start with:\n```\nuvicorn api.main:app --reload\n```")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Single Predict", "Batch Predict", "MLflow Runs"])

# ── Tab 1: Single Predict ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Single Tweet Classification")

    examples = [
        "My order hasn't arrived and the tracking shows no update for 5 days",
        "iPhone keeps restarting randomly after the latest iOS update",
        "My Uber driver cancelled and I was charged anyway",
        "My playlist won't load on mobile data but works on WiFi",
        "Flight was delayed 3 hours and I missed my connection",
        "Unexpected charge on my bill this month please help",
    ]

    example_choice = st.selectbox("Try an example:", ["(type your own)"] + examples)
    default_text = "" if example_choice == "(type your own)" else example_choice
    tweet = st.text_area("Enter customer tweet:", value=default_text, height=100,
                         placeholder="e.g. My package hasn't arrived...")

    if st.button("Classify", type="primary", use_container_width=True):
        if not tweet.strip():
            st.warning("Please enter a tweet.")
        else:
            with st.spinner("Classifying..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json={"text": tweet}, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()

                    company = result["predicted_company"]
                    conf = result["confidence"]
                    scores = result["all_scores"]
                    emoji = COMPANY_EMOJI.get(company, "🏢")

                    # Result card
                    st.markdown("---")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"### {emoji} {company}")
                        color = "green" if conf >= 0.6 else "orange" if conf >= 0.4 else "red"
                        st.markdown(f"**Confidence:** :{color}[{conf*100:.1f}%]")

                    with col2:
                        # Horizontal bar chart of all scores
                        labels = list(scores.keys())
                        values = [v * 100 for v in scores.values()]
                        colors = ["#1f77b4" if lbl != company else "#2ca02c" for lbl in labels]

                        fig = go.Figure(go.Bar(
                            x=values, y=labels, orientation="h",
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in values],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            xaxis_title="Confidence (%)",
                            xaxis=dict(range=[0, 105]),
                            margin=dict(l=0, r=40, t=10, b=10),
                            height=320,
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API. Make sure it is running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 2: Batch Predict ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Tweet Classification")
    st.markdown("Enter one tweet per line (max 100).")

    batch_input = st.text_area("Tweets (one per line):", height=200, placeholder=(
        "My order hasn't arrived\n"
        "iPhone keeps crashing after update\n"
        "Uber driver cancelled and I was charged"
    ))

    if st.button("Classify All", type="primary", use_container_width=True):
        texts = [t.strip() for t in batch_input.strip().splitlines() if t.strip()]
        if not texts:
            st.warning("Please enter at least one tweet.")
        elif len(texts) > 100:
            st.warning("Maximum 100 tweets at once.")
        else:
            with st.spinner(f"Classifying {len(texts)} tweets..."):
                try:
                    resp = requests.post(f"{API_URL}/predict/batch", json={"texts": texts}, timeout=30)
                    resp.raise_for_status()
                    results = resp.json()["results"]

                    rows = []
                    for r in results:
                        rows.append({
                            "Tweet": r["text"][:80] + ("..." if len(r["text"]) > 80 else ""),
                            "Predicted Company": f"{COMPANY_EMOJI.get(r['predicted_company'], '🏢')} {r['predicted_company']}",
                            "Confidence": f"{r['confidence']*100:.1f}%",
                        })

                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Distribution pie chart
                    company_counts = df["Predicted Company"].value_counts()
                    fig = go.Figure(go.Pie(
                        labels=company_counts.index,
                        values=company_counts.values,
                        hole=0.4,
                    ))
                    fig.update_layout(title="Distribution of Predicted Companies", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Download button
                    csv = pd.DataFrame(results).to_csv(index=False)
                    st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API. Make sure it is running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 3: MLflow Runs ────────────────────────────────────────────────────────
with tab3:
    st.subheader("MLflow Experiment Runs")

    try:
        import mlflow
        from pathlib import Path

        mlflow.set_tracking_uri(str(Path(__file__).parent / "mlruns"))
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("twitter-customer-support-classification")

        if exp:
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["metrics.accuracy DESC"],
            )

            rows = []
            for r in runs:
                rows.append({
                    "Run ID": r.info.run_id[:8],
                    "Model": r.data.params.get("model_type", "-"),
                    "Max Features": r.data.params.get("max_features", "-"),
                    "C": r.data.params.get("C", "-"),
                    "Accuracy": f"{r.data.metrics.get('accuracy', 0)*100:.2f}%",
                    "F1": f"{r.data.metrics.get('f1', 0)*100:.2f}%",
                    "Precision": f"{r.data.metrics.get('precision', 0)*100:.2f}%",
                    "Recall": f"{r.data.metrics.get('recall', 0)*100:.2f}%",
                    "Status": r.info.status,
                })

            df_runs = pd.DataFrame(rows)
            st.dataframe(df_runs, use_container_width=True, hide_index=True)

            # Accuracy comparison bar chart
            fig = go.Figure(go.Bar(
                x=[r["Run ID"] for r in rows],
                y=[float(r["Accuracy"].replace("%", "")) for r in rows],
                marker_color="#1f77b4",
                text=[r["Accuracy"] for r in rows],
                textposition="outside",
            ))
            fig.update_layout(
                title="Accuracy per Run",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[0, 105]),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No runs found. Run train_pipeline.py first.")

    except Exception as e:
        st.error(f"Could not load MLflow runs: {e}")
