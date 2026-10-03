import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

def evaluate_predictions(y_true: pd.Series, y_probs: np.ndarray, k_pcts=[0.01, 0.05, 0.10]) -> dict:
    """
    Computes comprehensive evaluation metrics for rare positive detection (money mules):
    - PR-AUC (Average Precision)
    - ROC-AUC
    - Recall @ Top K% riskiest accounts
    - PPP (Percentage of Population Reviewed to catch 90% of mules)
    """
    pr_auc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # Threshold at 0.5 for default stats
    y_pred_default = (y_probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_default, average="binary", zero_division=0)
    
    # Sort accounts by predicted risk score descending
    sorted_indices = np.argsort(y_probs)[::-1]
    sorted_y_true = np.array(y_true)[sorted_indices]
    total_mules = sorted_y_true.sum()
    total_accounts = len(y_true)
    
    recall_at_k = {}
    for k in k_pcts:
        top_n = max(int(total_accounts * k), 1)
        mules_caught = sorted_y_true[:top_n].sum()
        recall_at_k[f"Recall@Top_{int(k*100)}%"] = float(mules_caught / total_mules) if total_mules > 0 else 0.0
        
    # Compute PPP metric: % of population reviewed to achieve 90% recall
    if total_mules > 0:
        cum_mules = np.cumsum(sorted_y_true)
        target_90_mules = 0.90 * total_mules
        items_needed = np.searchsorted(cum_mules, target_90_mules) + 1
        ppp_90 = float(items_needed / total_accounts)
    else:
        ppp_90 = 0.0
        
    metrics = {
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "Precision": float(p),
        "Recall": float(r),
        "F1_Score": float(f1),
        "PPP_90%_Mules": ppp_90,
        **recall_at_k
    }
    return metrics

def run_early_window_experiment(df_raw: pd.DataFrame, labels_df: pd.DataFrame, windows=[1, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Evaluates how early a mule can be detected by varying the observation window size (in days).
    Returns a summary DataFrame comparing PR-AUC and Recall across observation windows.
    """
    from feature_engineering import build_early_and_graph_features
    from data_splitter import temporal_split_data
    from models import MuleDetectorModels
    
    results = []
    
    for w in windows:
        feats = build_early_and_graph_features(df_raw, window_days=w)
        X_train, y_train, _, _, X_test, y_test = temporal_split_data(feats, labels_df, df_raw)
        
        if len(X_train) == 0 or len(X_test) == 0 or y_train.sum() == 0 or y_test.sum() == 0:
            continue
            
        model_engine = MuleDetectorModels()
        model_engine.train_all(X_train, y_train)
        
        xgb_probs = model_engine.predict_proba("XGBoost", X_test)
        metrics = evaluate_predictions(y_test, xgb_probs)
        
        results.append({
            "Window_Days": w,
            "PR_AUC": metrics["PR_AUC"],
            "ROC_AUC": metrics["ROC_AUC"],
            "Recall@Top_5%": metrics.get("Recall@Top_5%", 0.0),
            "PPP_90%_Mules": metrics["PPP_90%_Mules"]
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.05, 0.15, 0.3, 0.0, 0.85, 0.02])
    print(evaluate_predictions(y_true, y_probs))
