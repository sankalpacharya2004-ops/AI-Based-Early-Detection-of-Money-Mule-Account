import os
import sys
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.generator import generate_synthetic_transactions
from src.label_builder import derive_account_labels
from src.feature_engineering import build_early_and_graph_features
from src.data_splitter import temporal_split_data
from src.models import MuleDetectorModels
from src.evaluation import evaluate_predictions, run_early_window_experiment
from src.explainability import MuleSHAPExplainer

def run_full_pipeline():
    print("=" * 70)
    print("AI-Based Early Detection of Money Mule Accounts - Pipeline Execution")
    print("=" * 70)
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Data Preparation / Synthetic Generation
    raw_data_path = "data/raw/HI-Small_Trans.csv"
    if os.path.exists(raw_data_path):
        print(f"\n[Step 1] Loading Kaggle IBM AML dataset from {raw_data_path}...")
        df_raw = pd.read_csv(raw_data_path)
    else:
        print("\n[Step 1] Kaggle IBM dataset not found in data/raw/. Generating synthetic transaction dataset...")
        df_raw = generate_synthetic_transactions(n_accounts=1000, n_transactions=25000, seed=42)
        sample_path = "data/synthetic_sample.csv"
        df_raw.to_csv(sample_path, index=False)
        print(f"-> Generated {len(df_raw)} transactions saved to {sample_path}")

    # Step 2: Account Labeling
    print("\n[Step 2] Deriving account-level mule proxy labels...")
    labels_df = derive_account_labels(df_raw)
    role_counts = labels_df['account_role'].value_counts().to_dict()
    print(f"-> Account Roles Breakdown: {role_counts}")
    
    # Step 3: Feature Engineering (7-Day Early Window)
    print("\n[Step 3] Building 7-Day early window & graph network features...")
    feats_df = build_early_and_graph_features(df_raw, window_days=7)
    print(f"-> Extracted {feats_df.shape[1]-1} features for {len(feats_df)} accounts.")
    
    # Save processed features & labels
    feats_df.to_csv("data/processed/account_features_7d.csv", index=False)
    labels_df.to_csv("data/processed/account_labels.csv", index=False)
    
    # Step 4: Temporal Split (No Temporal Leakage)
    print("\n[Step 4] Performing time-based temporal splitting (Train 60%, Val 20%, Test 20%)...")
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split_data(feats_df, labels_df, df_raw)
    print(f"-> Train: {len(X_train)} accounts (Mules: {y_train.sum()})")
    print(f"-> Val:   {len(X_val)} accounts (Mules: {y_val.sum()})")
    print(f"-> Test:  {len(X_test)} accounts (Mules: {y_test.sum()})")

    # Step 5: Model Training
    print("\n[Step 5] Training models (Logistic Regression, Random Forest, XGBoost, Isolation Forest)...")
    model_engine = MuleDetectorModels()
    models = model_engine.train_all(X_train, y_train)
    model_engine.save_models("models/mule_models.pkl")
    print("-> All models successfully trained and saved to models/mule_models.pkl")

    # Step 6: Model Evaluation & Benchmark Comparison
    print("\n[Step 6] Model Performance Comparison on Test Set:")
    eval_table = []
    for name in ["LogisticRegression", "RandomForest", "XGBoost", "IsolationForest"]:
        probs = model_engine.predict_proba(name, X_test)
        metrics = evaluate_predictions(y_test, probs)
        metrics["Model"] = name
        eval_table.append(metrics)
        
    df_eval = pd.DataFrame(eval_table)[["Model", "PR_AUC", "ROC_AUC", "Precision", "Recall", "F1_Score", "Recall@Top_5%", "PPP_90%_Mules"]]
    print(df_eval.to_string(index=False))
    df_eval.to_csv("data/processed/model_evaluation_metrics.csv", index=False)

    # Step 7: Early Detection Window Sensitivity Experiment
    print("\n[Step 7] Running Early Detection Window Sensitivity Analysis (1, 3, 7, 14, 30 Days)...")
    window_df = run_early_window_experiment(df_raw, labels_df, windows=[1, 3, 7, 14, 30])
    print(window_df.to_string(index=False))
    window_df.to_csv("data/processed/window_experiment_results.csv", index=False)

    # Step 8: Model Explainability (SHAP)
    print("\n[Step 8] Computing SHAP Global Feature Importance...")
    try:
        xgb_model = models["XGBoost"]
        shap_engine = MuleSHAPExplainer(xgb_model, X_train)
        importance_df = shap_engine.get_global_importance(X_test)
        importance_df.to_csv("data/processed/shap_global_importance.csv", index=False)
        print("-> Top 5 Predictive Features according to SHAP:")
        print(importance_df.head(5).to_string(index=False))
    except Exception as e:
        print(f"-> SHAP computation warning: {e}")

    print("\n" + "=" * 70)
    print("Pipeline completed successfully! All artifacts stored in data/processed/ and models/")
    print("=" * 70)

if __name__ == "__main__":
    run_full_pipeline()
