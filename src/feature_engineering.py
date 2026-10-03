import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta

def build_early_and_graph_features(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Computes early-window behavioral features (first N days of account life) 
    and graph network metrics (NetworkX) without target or future temporal leakage.
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    # 1. Expand transactions into an account-centric view (both incoming & outgoing records)
    out_records = df[["Account", "Timestamp", "Amount Paid", "Payment Format", "To Bank", "Account.1"]].copy()
    out_records.columns = ["acc", "time", "amt", "format", "counterparty_bank", "counterparty_acc"]
    out_records["direction"] = "out"
    
    in_records = df[["Account.1", "Timestamp", "Amount Received", "Payment Format", "From Bank", "Account"]].copy()
    in_records.columns = ["acc", "time", "amt", "format", "counterparty_bank", "counterparty_acc"]
    in_records["direction"] = "in"
    
    tx_long = pd.concat([out_records, in_records], ignore_index=True)
    
    # Calculate each account's first seen timestamp (account "birth")
    first_seen = tx_long.groupby("acc")["time"].min().rename("first_seen")
    tx_long = tx_long.join(first_seen, on="acc")
    
    # Account age in days at transaction time
    tx_long["age_days"] = (tx_long["time"] - tx_long["first_seen"]).dt.total_seconds() / 86400.0
    
    # Filter to transactions occurring strictly within the observation window (e.g., first N days)
    early_tx = tx_long[tx_long["age_days"] <= window_days].copy()
    
    # 2. Compute Account-Level Window Features
    grouped = early_tx.groupby("acc")
    
    feature_list = []
    
    for acc, group in grouped:
        in_group = group[group["direction"] == "in"]
        out_group = group[group["direction"] == "out"]
        
        n_tx = len(group)
        n_in = len(in_group)
        n_out = len(out_group)
        
        total_in = in_group["amt"].sum() if n_in > 0 else 0.0
        total_out = out_group["amt"].sum() if n_out > 0 else 0.0
        
        max_amt = group["amt"].max() if n_tx > 0 else 0.0
        mean_amt = group["amt"].mean() if n_tx > 0 else 0.0
        std_amt = group["amt"].std() if n_tx > 1 else 0.0
        
        # Velocity / Delay: time from first inflow to first outflow in hours
        if n_in > 0 and n_out > 0:
            first_in_t = in_group["time"].min()
            first_out_t = out_group["time"].min()
            delay_hrs = max((first_out_t - first_in_t).total_seconds() / 3600.0, 0.0)
        else:
            delay_hrs = 999.0  # Default high value if no pass-through yet
            
        # Pass-through & Balance drainage
        pass_through_ratio = total_out / (total_in + 1.0)
        in_out_balance_ratio = abs(total_in - total_out) / (total_in + total_out + 1.0)
        
        # Counterparty diversity
        distinct_senders = in_group["counterparty_acc"].nunique()
        distinct_receivers = out_group["counterparty_acc"].nunique()
        
        # Structuring check ($8,500 - $9,999 transactions)
        structuring_cnt = ((group["amt"] >= 8500) & (group["amt"] <= 9999)).sum()
        structuring_ratio = structuring_cnt / n_tx if n_tx > 0 else 0.0
        
        # Format distribution
        wire_cnt = (group["format"].isin(["Wire", "ACH"])).sum()
        wire_ratio = wire_cnt / n_tx if n_tx > 0 else 0.0
        
        feature_list.append({
            "account_id": acc,
            "n_tx": n_tx,
            "n_in": n_in,
            "n_out": n_out,
            "total_in": total_in,
            "total_out": total_out,
            "max_amt": max_amt,
            "mean_amt": mean_amt,
            "std_amt": std_amt,
            "delay_first_in_out_hrs": delay_hrs,
            "pass_through_ratio": pass_through_ratio,
            "in_out_balance_ratio": in_out_balance_ratio,
            "distinct_senders": distinct_senders,
            "distinct_receivers": distinct_receivers,
            "structuring_ratio": structuring_ratio,
            "wire_ratio": wire_ratio
        })
        
    feats_df = pd.DataFrame(feature_list)
    
    # 3. Compute Network Graph Features using NetworkX
    # Build Directed Graph from transactions in early window
    G = nx.DiGraph()
    for _, row in early_tx[early_tx["direction"] == "out"].iterrows():
        G.add_edge(row["acc"], row["counterparty_acc"], weight=row["amt"])
        
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    try:
        pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        pagerank_scores = {node: 0.0 for node in G.nodes()}
        
    graph_feats = []
    for acc in feats_df["account_id"]:
        in_deg = in_degrees.get(acc, 0)
        out_deg = out_degrees.get(acc, 0)
        pr = pagerank_scores.get(acc, 0.0)
        deg_ratio = out_deg / (in_deg + 1.0)
        
        graph_feats.append({
            "account_id": acc,
            "graph_in_degree": in_deg,
            "graph_out_degree": out_deg,
            "graph_degree_ratio": deg_ratio,
            "graph_pagerank": pr
        })
        
    graph_df = pd.DataFrame(graph_feats)
    
    # Merge behavioral features + graph features
    final_features = pd.merge(feats_df, graph_df, on="account_id", how="left").fillna(0)
    return final_features

if __name__ == "__main__":
    from generator import generate_synthetic_transactions
    df = generate_synthetic_transactions()
    feats = build_early_and_graph_features(df, window_days=7)
    print(f"Engineered features for {len(feats)} accounts.")
    print(feats.head())
