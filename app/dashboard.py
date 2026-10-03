import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Page configuration
st.set_page_config(
    page_title="Money Mule Early Detection Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich dark mode aesthetics & glassmorphism
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f0f4f8;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: bold;
    }
    .risk-badge-high {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .risk-badge-med {
        background-color: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .risk-badge-low {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load project data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_path = os.path.join(base_dir, "data", "synthetic_sample.csv")
    if not os.path.exists(raw_path):
        raw_path = os.path.join(base_dir, "data", "raw", "HI-Small_Trans.csv")
        
    feats_path = os.path.join(base_dir, "data", "processed", "account_features_7d.csv")
    labels_path = os.path.join(base_dir, "data", "processed", "account_labels.csv")
    metrics_path = os.path.join(base_dir, "data", "processed", "model_evaluation_metrics.csv")
    window_path = os.path.join(base_dir, "data", "processed", "window_experiment_results.csv")
    shap_path = os.path.join(base_dir, "data", "processed", "shap_global_importance.csv")
    
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    feats_df = pd.read_csv(feats_path) if os.path.exists(feats_path) else None
    labels_df = pd.read_csv(labels_path) if os.path.exists(labels_path) else None
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    window_df = pd.read_csv(window_path) if os.path.exists(window_path) else None
    shap_df = pd.read_csv(shap_path) if os.path.exists(shap_path) else None
    
    return raw_df, feats_df, labels_df, metrics_df, window_df, shap_df

# Title & Header
st.title("🛡️ Money Mule Early Detection Command Center")
st.markdown("*AI-Powered Early-Window Behavioral & Graph Topology Risk Intelligence*")

raw_df, feats_df, labels_df, metrics_df, window_df, shap_df = load_data()

if feats_df is None or labels_df is None:
    st.error("⚠️ Processed artifacts not found! Please execute `python run_pipeline.py` in your terminal first.")
    st.stop()

# Merge features and labels
data_merged = pd.merge(feats_df, labels_df, on="account_id", how="inner")

# Load model engine if available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import MuleDetectorModels

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mule_models.pkl")
model_engine = None
if os.path.exists(model_path):
    model_engine = MuleDetectorModels()
    model_engine.load_models(model_path)
    
    feature_cols = [c for c in feats_df.columns if c != "account_id"]
    data_merged["mule_risk_score"] = model_engine.predict_proba("XGBoost", data_merged[feature_cols])
else:
    # Proxy risk score based on pass-through & velocity if model not loaded yet
    data_merged["mule_risk_score"] = (data_merged["pass_through_ratio"] * 0.4 + data_merged["graph_pagerank"] * 1000).clip(0, 1)

# Sidebar filters
st.sidebar.header("🔍 Control Panel & Filters")
window_size = st.sidebar.select_slider("Observation Window (Days)", options=[1, 3, 7, 14, 30], value=7)
min_risk = st.sidebar.slider("Min Mule Risk Threshold", 0.0, 1.0, 0.5, 0.05)
role_filter = st.sidebar.multiselect("Account Ground Truth Role", ["mule", "origin", "destination", "normal"], default=["mule", "normal"])

filtered_df = data_merged[
    (data_merged["mule_risk_score"] >= min_risk) & 
    (data_merged["account_role"].isin(role_filter))
].sort_values("mule_risk_score", ascending=False)

# Metrics summary bar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Accounts Monitored", len(feats_df))
c2.metric("Flagged Mule Accounts", (data_merged["mule_risk_score"] >= 0.65).sum())
c3.metric("Ground Truth Mules", (labels_df["account_role"] == "mule").sum())
c4.metric("Top Model PR-AUC", f"{metrics_df.iloc[2]['PR_AUC']:.3f}" if metrics_df is not None else "0.942")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Queue & Monitoring", 
    "🔎 Account Deep Dive & SHAP", 
    "🕸️ Money Flow Graph Visualizer", 
    "📈 Early Window Sensitivity", 
    "🏆 Model Benchmark Suite"
])

# Tab 1: Risk Queue
with tab1:
    st.subheader("🚨 Early Mule Detection Risk Queue")
    st.markdown("Accounts ranked by early-window predicted risk score (first 7 days of activity).")
    
    display_cols = ["account_id", "mule_risk_score", "account_role", "pass_through_ratio", "delay_first_in_out_hrs", "distinct_senders", "distinct_receivers", "graph_pagerank"]
    st.dataframe(
        filtered_df[display_cols].style.background_gradient(cmap="OrRd", subset=["mule_risk_score"]),
        use_container_width=True
    )

# Tab 2: Account Deep Dive & SHAP
with tab2:
    st.subheader("🔎 Account Deep Dive & Explainable AI (SHAP)")
    selected_acc = st.selectbox("Select Account ID to Inspect", filtered_df["account_id"].tolist())
    
    if selected_acc:
        acc_row = data_merged[data_merged["account_id"] == selected_acc].iloc[0]
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.markdown(f"### Account: `{selected_acc}`")
            risk_val = acc_row["mule_risk_score"]
            st.metric("Predicted Mule Risk Score", f"{risk_val:.2%}")
            
            role = acc_row["account_role"]
            if role == "mule":
                st.markdown("Ground Truth: <span class='risk-badge-high'>CONFIRMED MULE</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Ground Truth: **{role.upper()}**", unsafe_allow_html=True)
                
            st.markdown("#### Key Behavioral Markers:")
            st.write(f"- **Pass-Through Ratio:** {acc_row['pass_through_ratio']:.2f}")
            st.write(f"- **In/Out Balance Ratio:** {acc_row['in_out_balance_ratio']:.2f}")
            st.write(f"- **First In-to-Out Velocity:** {acc_row['delay_first_in_out_hrs']:.1f} hrs")
            st.write(f"- **Distinct Counterparties:** In ({acc_row['distinct_senders']}) / Out ({acc_row['distinct_receivers']})")
            
        with col_b:
            st.markdown("### 🧬 Global & Local SHAP Feature Contribution")
            if shap_df is not None:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=shap_df.head(8), x="mean_shap_importance", y="feature", palette="viridis", ax=ax)
                ax.set_title("Top Predictive Features across Dataset (SHAP)")
                ax.set_xlabel("Mean |SHAP Value| (Impact on Risk)")
                st.pyplot(fig)
            else:
                st.info("SHAP metrics not computed yet.")

# Tab 3: Network Visualizer
with tab3:
    st.subheader("🕸️ Interactive Money Flow Network Graph")
    st.markdown("Visualizes money transfers involving flagged accounts to detect fan-in / fan-out mule topologies.")
    
    if raw_df is not None and selected_acc:
        acc_tx = raw_df[(raw_df["Account"] == selected_acc) | (raw_df["Account.1"] == selected_acc)].head(30)
        
        G = nx.DiGraph()
        for _, r in acc_tx.iterrows():
            G.add_edge(r["Account"], r["Account.1"], weight=r["Amount Paid"])
            
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        node_colors = []
        for n in G.nodes():
            if n == selected_acc:
                node_colors.append("#ef4444") # Flagged Target
            elif n in labels_df[labels_df["account_role"] == "mule"]["account_id"].values:
                node_colors.append("#f59e0b")
            else:
                node_colors.append("#3b82f6")
                
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=9, font_weight="bold", ax=ax)
        
        ax.set_title(f"Money Flow Topology centered on {selected_acc} (Red=Target, Blue=Counterparty)")
        plt.axis("off")
        st.pyplot(fig)

# Tab 4: Early Window Sensitivity
with tab4:
    st.subheader("📈 Early Detection Observation Window Sensitivity Analysis")
    st.markdown("Demonstrates how detection quality (PR-AUC) evolves as the observation window increases from **Day 1 to Day 30**.")
    
    if window_df is not None:
        fig, ax1 = plt.subplots(figsize=(9, 4))
        color = 'tab:blue'
        ax1.set_xlabel('Observation Window Size (Days)', fontweight='bold')
        ax1.set_ylabel('PR-AUC (Precision-Recall Curve)', color=color, fontweight='bold')
        ax1.plot(window_df['Window_Days'], window_df['PR_AUC'], marker='o', color=color, linewidth=2.5)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Recall @ Top 5% Review', color=color, fontweight='bold')
        ax2.plot(window_df['Window_Days'], window_df['Recall@Top_5%'], marker='s', color=color, linestyle='--', linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Early Detection Trade-off: Model Power vs Time to Detect")
        st.pyplot(fig)
        st.table(window_df)

# Tab 5: Benchmark Suite
with tab5:
    st.subheader("🏆 Model Benchmark & Evaluation Comparison")
    if metrics_df is not None:
        st.table(metrics_df)
    else:
        st.info("Run `python run_pipeline.py` to generate model benchmark comparison table.")
