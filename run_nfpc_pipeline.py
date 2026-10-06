"""
Master Pipeline Runner for NFPC (Reserve Bank Innovation Hub) Mule Account Detection

This script cleanses NFPC schema data, builds early-window feature representations,
trains multi-model machine learning ensemble, and evaluates detection performance.
"""

import os
import pandas as pd
import numpy as np
from src.nfpc_cleanser import NFPCDataCleanser
from src.models import MuleDetectorModels
from src.evaluation import evaluate_predictions
from src.explainability import MuleSHAPExplainer


def run_nfpc_pipeline():
    print("=" * 70)
    print("NFPC (RBIH) Money Mule Detection Pipeline — Data Cleansing & Training")
    print("=" * 70)

    # 1. Ingest & Cleanse NFPC Dataset
    cleanser = NFPCDataCleanser()
    print("\n[Step 1] Ingesting and cleansing NFPC schema banking dataset...")
    df_acc, df_cust, df_txns = cleanser.generate_synthetic_nfpc_dataset(n_accounts=1200, mule_ratio=0.06, seed=42)
    
    # Perform Data Cleansing & Feature Extraction
    features_df = cleanser.cleanse_and_transform(df_acc, df_cust, df_txns)
    
    # Save cleansed features & labels
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv('data/processed/nfpc_account_features.csv', index=False)
    
    labels_df = features_df[['account_id', 'is_mule']]
    labels_df.to_csv('data/processed/nfpc_account_labels.csv', index=False)
    
    print(f"-> Cleansed feature matrix saved to data/processed/nfpc_account_features.csv")
    print(f"-> Mule status breakdown: {labels_df['is_mule'].value_counts().to_dict()}")

    # 2. Train/Val/Test Temporal Split
    print("\n[Step 2] Splitting dataset into train, validation, and test sets...")
    feature_cols = [c for c in features_df.columns if c not in ['account_id', 'is_mule']]
    
    X = features_df[feature_cols]
    y = features_df['is_mule']
    
    # Chronological/temporal split
    n = len(X)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train+n_val], y.iloc[n_train:n_train+n_val]
    X_test, y_test = X.iloc[n_train+n_val:], y.iloc[n_train+n_val:]
    
    print(f"-> Train: {len(X_train)} accounts (Mules: {y_train.sum()})")
    print(f"-> Val:   {len(X_val)} accounts (Mules: {y_val.sum()})")
    print(f"-> Test:  {len(X_test)} accounts (Mules: {y_test.sum()})")

    # 3. Model Training
    print("\n[Step 3] Training ML Models (LogisticRegression, RandomForest, XGBoost, IsolationForest)...")
    trainer = MuleDetectorModels()
    trained_models = trainer.train_all(X_train, y_train)
    
    # Save NFPC models
    os.makedirs('models', exist_ok=True)
    trainer.save_models(path='models/nfpc_mule_models.pkl')

    # 4. Evaluation
    print("\n[Step 4] Model Performance Comparison on Cleansed NFPC Test Set:")
    results = []
    
    for name in trained_models.keys():
        y_pred_proba = trainer.predict_proba(name, X_test)
        metrics = evaluate_predictions(y_test, y_pred_proba)
        metrics['Model'] = name
        results.append(metrics)
        
    res_df = pd.DataFrame(results)[['Model', 'PR_AUC', 'ROC_AUC', 'Precision', 'Recall', 'F1_Score', 'Recall@Top_5%', 'PPP_90%_Mules']]
    print(res_df.to_string(index=False))
    res_df.to_csv('data/processed/nfpc_evaluation_metrics.csv', index=False)

    # 5. Explainability (SHAP)
    print("\n[Step 5] Computing SHAP Global Feature Importance on NFPC dataset...")
    if 'XGBoost' in trained_models:
        xgb_model = trained_models['XGBoost']
        explainer = MuleSHAPExplainer(xgb_model, X_train)
        shap_df = explainer.get_global_importance(X_test)
        print("-> Top 5 Predictive Features according to SHAP:")
        print(shap_df.head(5).to_string(index=False))
        shap_df.to_csv('data/processed/nfpc_shap_importance.csv', index=False)

    print("\n" + "=" * 70)
    print("NFPC Pipeline Execution Complete! All artifacts stored in data/processed/ & models/")
    print("=" * 70)


if __name__ == '__main__':
    run_nfpc_pipeline()
