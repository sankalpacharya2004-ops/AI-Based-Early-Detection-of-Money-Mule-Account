"""
NFPC (National Fraud Prevention Challenge - RBIH) Data Cleanser & Adapter

This module ingests, cleanses, standardizes, and featurizes Indian banking data 
(NFPC Phase 1 & 2 schema) for early money mule detection.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class NFPCDataCleanser:
    def __init__(self, ref_date='2025-06-30'):
        self.ref_date = pd.to_datetime(ref_date)
        
    def generate_synthetic_nfpc_dataset(self, n_accounts=1000, mule_ratio=0.05, seed=42):
        """
        Generates a synthetic NFPC schema dataset adhering to RBIH challenge specifications.
        """
        np.random.seed(seed)
        n_mules = int(n_accounts * mule_ratio)
        n_legit = n_accounts - n_mules
        
        account_ids = [f"ACCT_{i:06d}" for i in range(n_accounts)]
        is_mule = np.array([1] * n_mules + [0] * n_legit)
        np.random.shuffle(is_mule)
        
        # 1. Generate Accounts & Customers
        start_date = datetime(2023, 1, 1)
        account_rows = []
        customer_rows = []
        txn_rows = []
        
        channels = ['UPC', 'UPD', 'IPM', 'NTD', 'FTD', 'ATW', 'CHQ', 'END', 'P2A', 'MCR']
        mccs = [6011, 5933, 6051, 6012, 4814, 5411, 5812, 5311, 7995]
        
        for idx, acct_id in enumerate(account_ids):
            mule_flag = is_mule[idx]
            cust_id = f"CUST_{idx:06d}"
            
            op_date = start_date + timedelta(days=int(np.random.randint(0, 500)))
            acct_status = 'frozen' if (mule_flag and np.random.rand() > 0.3) else 'active'
            freeze_date = op_date + timedelta(days=int(np.random.randint(5, 30))) if acct_status == 'frozen' else pd.NaT
            
            avg_bal = np.random.uniform(500, 50000) if not mule_flag else np.random.uniform(10, 2000)
            
            account_rows.append({
                'account_id': acct_id,
                'customer_id': cust_id,
                'account_status': acct_status,
                'account_opening_date': op_date.strftime('%Y-%m-%d'),
                'avg_balance': avg_bal,
                'monthly_avg_balance': avg_bal * np.random.uniform(0.9, 1.1),
                'quarterly_avg_balance': avg_bal * np.random.uniform(0.85, 1.15),
                'daily_avg_balance': avg_bal * np.random.uniform(0.8, 1.2),
                'freeze_date': freeze_date.strftime('%Y-%m-%d') if pd.notna(freeze_date) else None,
                'product_family': np.random.choice(['S', 'K', 'O']),
                'kyc_compliant': 'Y' if (not mule_flag or np.random.rand() > 0.4) else 'N',
                'rural_branch': 'Y' if np.random.rand() > 0.7 else 'N',
                'is_mule': mule_flag
            })
            
            customer_rows.append({
                'customer_id': cust_id,
                'dob': (start_date - timedelta(days=int(np.random.randint(18*365, 65*365)))).strftime('%Y-%m-%d'),
                'pan_available': 'Y' if (not mule_flag or np.random.rand() > 0.2) else 'N',
                'aadhaar_available': 'Y' if np.random.rand() > 0.3 else 'N',
                'mobile_banking_flag': 'Y' if np.random.rand() > 0.4 else 'N',
                'internet_banking_flag': 'Y' if np.random.rand() > 0.4 else 'N',
            })
            
            # Generate Transactions (within first 7-14 days)
            n_txns = np.random.randint(5, 25) if mule_flag else np.random.randint(2, 10)
            for t_idx in range(n_txns):
                t_time = op_date + timedelta(hours=int(np.random.randint(1, 168)))
                
                if mule_flag:
                    # Mule pattern: rapid pass-through, structuring near 50k, high channel diversity
                    is_credit = (t_idx % 2 == 0)
                    amt = np.random.choice([49500, 48000, 49000, 10000, 25000]) if np.random.rand() > 0.4 else np.random.uniform(5000, 50000)
                    ch = np.random.choice(['UPC', 'UPD', 'FTD', 'P2A'])
                    mcc = np.random.choice([6011, 6051, 4814, 5933])
                    cp_id = f"CP_{np.random.randint(1, 50):05d}"
                else:
                    is_credit = (np.random.rand() > 0.6)
                    amt = np.random.uniform(200, 8000)
                    ch = np.random.choice(channels)
                    mcc = np.random.choice(mccs)
                    cp_id = f"CP_{np.random.randint(100, 500):05d}"
                    
                txn_rows.append({
                    'transaction_id': f"TXN_{len(txn_rows):08d}",
                    'account_id': acct_id,
                    'transaction_timestamp': t_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'mcc_code': mcc,
                    'channel': ch,
                    'amount': amt,
                    'txn_type': 'C' if is_credit else 'D',
                    'counterparty_id': cp_id
                })
                
        df_accounts = pd.DataFrame(account_rows)
        df_customers = pd.DataFrame(customer_rows)
        df_txns = pd.DataFrame(txn_rows)
        
        return df_accounts, df_customers, df_txns

    def cleanse_and_transform(self, df_accounts, df_customers, df_txns):
        """
        Cleanses raw NFPC tables, handles nulls, normalizes features, and builds account feature matrix.
        """
        print("Cleansing & Parsing NFPC Data...")
        
        # 1. Parse Dates safely
        df_accounts['account_opening_date'] = pd.to_datetime(df_accounts['account_opening_date'], errors='coerce')
        df_txns['transaction_timestamp'] = pd.to_datetime(df_txns['transaction_timestamp'], errors='coerce')
        
        # 2. Handle missing values
        for col in ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']:
            if col in df_accounts.columns:
                df_accounts[col] = df_accounts[col].fillna(df_accounts[col].median() if not df_accounts[col].dropna().empty else 0.0)
                
        # 3. Categorical encoding & indicators
        df_accounts['was_frozen'] = df_accounts['freeze_date'].notna().astype(int) if 'freeze_date' in df_accounts.columns else 0
        df_accounts['acct_age_days'] = (self.ref_date - df_accounts['account_opening_date']).dt.days.fillna(30)
        
        # 4. Feature Extraction from Transactions
        print("Extracting Early-Window Behavioral & Network Features...")
        
        # Sort transactions
        df_txns = df_txns.sort_values(['account_id', 'transaction_timestamp'])
        
        # Basic Aggregations
        agg_txns = df_txns.groupby('account_id').agg(
            n_txns=('transaction_id', 'count'),
            total_amt=('amount', 'sum'),
            mean_amt=('amount', 'mean'),
            max_amt=('amount', 'max'),
            std_amt=('amount', 'std'),
            unique_cp=('counterparty_id', 'nunique'),
            unique_channels=('channel', 'nunique')
        ).reset_index().fillna(0)
        
        # Credit vs Debit
        credits = df_txns[df_txns['txn_type'] == 'C'].groupby('account_id')['amount'].agg(
            n_in='count', total_in='sum'
        ).reset_index()
        
        debits = df_txns[df_txns['txn_type'] == 'D'].groupby('account_id')['amount'].agg(
            n_out='count', total_out='sum'
        ).reset_index()
        
        # Structuring signals (Transactions close to 50k INR threshold)
        structuring = df_txns.groupby('account_id')['amount'].apply(
            lambda x: ((x >= 45000) & (x < 50000)).mean()
        ).reset_index().rename(columns={'amount': 'structuring_ratio'})
        
        # High Risk Channel Usage (UPC/UPD/FTD)
        high_risk_ch = df_txns.groupby('account_id')['channel'].apply(
            lambda x: (x.isin(['UPC', 'UPD', 'FTD', 'P2A'])).mean()
        ).reset_index().rename(columns={'channel': 'high_risk_channel_ratio'})
        
        # Merge all components
        features = df_accounts[['account_id', 'avg_balance', 'acct_age_days', 'was_frozen', 'is_mule']].merge(
            agg_txns, on='account_id', how='left'
        ).merge(
            credits, on='account_id', how='left'
        ).merge(
            debits, on='account_id', how='left'
        ).merge(
            structuring, on='account_id', how='left'
        ).merge(
            high_risk_ch, on='account_id', how='left'
        ).fillna(0)
        
        # Derived Ratios
        features['pass_through_ratio'] = features['total_out'] / (features['total_in'] + 1e-5)
        features['in_out_balance_ratio'] = (features['total_in'] - features['total_out']) / (features['total_in'] + 1.0)
        features['cp_per_txn'] = features['unique_cp'] / (features['n_txns'] + 1.0)
        
        # Outlier Clipping & Normalization
        features['mean_amt'] = np.log10(np.maximum(features['mean_amt'], 1.0))
        features['total_in'] = np.log10(np.maximum(features['total_in'], 1.0))
        features['total_out'] = np.log10(np.maximum(features['total_out'], 1.0))
        
        print(f"Cleansing completed: {len(features)} account features processed.")
        return features


if __name__ == '__main__':
    cleanser = NFPCDataCleanser()
    df_acc, df_cust, df_txn = cleanser.generate_synthetic_nfpc_dataset(n_accounts=500)
    features = cleanser.cleanse_and_transform(df_acc, df_cust, df_txn)
    print(features.head())
