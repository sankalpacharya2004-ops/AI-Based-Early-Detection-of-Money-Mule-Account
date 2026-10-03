# AI-Based Early Detection of Money Mule Accounts

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit%20%7C%20XGBoost-orange)](https://github.com/)

An end-to-end Machine Learning and Graph Analytics framework designed to detect **Money Mule accounts early in their lifecycle** (first 1–7 days) before significant illicit money is laundered.

---

## 📌 Project Highlights & Research Novelty

Traditional anti-money laundering (AML) models flag transactions **after** laundering has occurred. This project introduces an **Early-Window Account Detection Architecture** that:
1. **Prevents Temporal Leakage**: Employs temporal time-based splits (Jullum et al.) rather than random splits to ensure no future information leaks into training.
2. **Derived Mule Proxy Ground Truth**: Defines mule accounts as pass-through entities (accounts receiving dirty money and rapidly forwarding it).
3. **Early Behavioral & Graph Features**: Computes rolling pass-through velocity, balance drainage, counterparty diversity, structuring signals, and NetworkX PageRank/Degree ratios strictly within the account's first $N$ days of existence.
4. **Hard Negative Ingestion**: Integrates high-volume non-mule commercial accounts to optimize model precision.
5. **Explainable AI (SHAP)**: Provides feature-level auditability for compliance investigators.

---

## 🛠️ Architecture & Folder Structure

```
money mule project/
├── data/
│   ├── raw/                      # Raw dataset directory (e.g. Kaggle IBM AML HI-Small_Trans.csv)
│   ├── processed/                # Processed feature tables, labels, metrics, SHAP results
│   └── synthetic_sample.csv       # Auto-generated synthetic transactions for out-of-the-box demo
├── src/
│   ├── generator.py              # Realistic IBM AML schema synthetic transaction generator
│   ├── label_builder.py          # Account-level ground truth label derivation (Mule vs Normal)
│   ├── feature_engineering.py    # Early-window behavioral velocity & graph network features
│   ├── data_splitter.py          # Temporal time-based train/val/test splitter
│   ├── models.py                 # LogisticRegression, RandomForest, XGBoost, IsolationForest
│   ├── evaluation.py             # PR-AUC, Recall@K, PPP metrics & early-window sensitivity experiment
│   └── explainability.py         # SHAP TreeExplainer for global & local feature attribution
├── app/
│   ├── api.py                    # FastAPI REST endpoint (/score, /explain, /health)
│   └── dashboard.py              # Streamlit interactive command center & graph visualizer
├── models/                       # Saved trained model artifacts (.pkl)
├── run_pipeline.py               # One-click CLI master pipeline execution script
├── requirements.txt              # System dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
Executes data generation, feature engineering, model training, SHAP explainability, and sensitivity analysis:
```bash
python run_pipeline.py
```

### 3. Launch the Streamlit Dashboard
Launch the interactive visual command center:
```bash
python -m streamlit run app/dashboard.py
```
*Access the dashboard at `http://localhost:8501`*

### 4. Launch the FastAPI REST Service
Start the real-time scoring and explanation API:
```bash
python -m uvicorn app.api:app --reload
```
*Access API docs at `http://127.0.0.1:8000/docs`*

---

## 📊 Evaluation & Metrics Summary

- **Primary Metric**: PR-AUC (Average Precision Score) for extreme class imbalance.
- **Recall@Top-5%**: Catch rate when compliance officers review top 5% highest risk accounts.
- **PPP (Predicted Positive Proportion)**: Percentage of population reviewed to achieve 90% mule detection.

---

## 📜 References
- **Jullum et al. (DNB Bank)**: *Machine learning for anti-money laundering: a practical perspective*.
- **Li et al. (China CBDC)**: *Anti-Money Laundering in Central Bank Digital Currency via Graph Neural Networks*.
- **IBM AML Synthetic Dataset**: *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*.
