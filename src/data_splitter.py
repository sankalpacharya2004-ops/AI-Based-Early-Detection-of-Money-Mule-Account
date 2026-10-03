import pandas as pd
import numpy as np
from typing import Tuple

def temporal_split_data(
    features_df: pd.DataFrame, 
    labels_df: pd.DataFrame, 
    transactions_df: pd.DataFrame,
    train_pct: float = 0.60,
    val_pct: float = 0.20
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Performs time-based temporal splitting based on account creation (first transaction time).
    Prevents temporal data leakage and cheating in fraud/mule detection models.
    """
    tx = transactions_df.copy()
    tx["Timestamp"] = pd.to_datetime(tx["Timestamp"])
    
    # Compute account first seen time
    first_seen_from = tx.groupby("Account")["Timestamp"].min()
    first_seen_to = tx.groupby("Account.1")["Timestamp"].min()
    first_seen = pd.concat([first_seen_from, first_seen_to]).groupby(level=0).min()
    
    # Merge labels with features
    df_merged = pd.merge(features_df, labels_df[['account_id', 'is_mule']], on="account_id", how="inner")
    df_merged["first_seen"] = df_merged["account_id"].map(first_seen)
    
    # Sort by first seen timestamp
    df_merged = df_merged.sort_values("first_seen").reset_index(drop=True)
    
    # Quantiles for cutoff times
    cut1 = df_merged["first_seen"].quantile(train_pct)
    cut2 = df_merged["first_seen"].quantile(train_pct + val_pct)
    
    train_mask = df_merged["first_seen"] <= cut1
    val_mask = (df_merged["first_seen"] > cut1) & (df_merged["first_seen"] <= cut2)
    test_mask = df_merged["first_seen"] > cut2
    
    feature_cols = [c for c in features_df.columns if c != "account_id"]
    
    X_train = df_merged.loc[train_mask, feature_cols]
    y_train = df_merged.loc[train_mask, "is_mule"]
    
    X_val = df_merged.loc[val_mask, feature_cols]
    y_val = df_merged.loc[val_mask, "is_mule"]
    
    X_test = df_merged.loc[test_mask, feature_cols]
    y_test = df_merged.loc[test_mask, "is_mule"]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    from generator import generate_synthetic_transactions
    from label_builder import derive_account_labels
    from feature_engineering import build_early_and_graph_features
    
    df = generate_synthetic_transactions()
    labels = derive_account_labels(df)
    feats = build_early_and_graph_features(df)
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = temporal_split_data(feats, labels, df)
    print(f"Train size: {len(X_tr)} (Mules: {y_tr.sum()})")
    print(f"Val size:   {len(X_val)} (Mules: {y_val.sum()})")
    print(f"Test size:  {len(X_te)} (Mules: {y_te.sum()})")
