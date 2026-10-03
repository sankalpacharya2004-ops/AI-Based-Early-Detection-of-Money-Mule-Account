import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_transactions(n_accounts: int = 1000, n_transactions: int = 25000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic financial transaction dataset matching the IBM AML HI-Small schema.
    Simulates normal commercial activity + money mule behaviors (fan-in/fan-out, pass-through, structuring).
    """
    np.random.seed(seed)
    
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    banks = ["Bank_Alpha", "Bank_Beta", "Bank_Gamma", "Bank_Delta", "Bank_Echo"]
    currencies = ["USD", "EUR", "GBP"]
    formats = ["ACH", "Wire", "Credit Card", "Cheque", "Cash"]
    
    accounts = [f"ACC_{i:05d}" for i in range(1, n_accounts + 1)]
    
    # Designate roles for ground truth (used for simulation)
    n_mules = int(n_accounts * 0.05)       # 5% Mules
    n_origins = int(n_accounts * 0.03)     # 3% Criminal Senders
    n_destinations = int(n_accounts * 0.02)# 2% Cashout Destinations
    
    mule_accs = set(accounts[:n_mules])
    origin_accs = set(accounts[n_mules : n_mules + n_origins])
    dest_accs = set(accounts[n_mules + n_origins : n_mules + n_origins + n_destinations])
    normal_accs = set(accounts[n_mules + n_origins + n_destinations:])
    
    # Assign each account a creation date (birth timestamp) evenly spread across 50 days
    account_birthdays = {}
    for acc in accounts:
        birth_day = np.random.uniform(0, 50)
        account_birthdays[acc] = start_date + timedelta(days=birth_day)
        
    records = []
    
    # 1. Normal Transactions (Routine commercial activity)
    for _ in range(int(n_transactions * 0.82)):
        sender = np.random.choice(list(normal_accs))
        receiver = np.random.choice(list(normal_accs))
        while receiver == sender:
            receiver = np.random.choice(list(normal_accs))
            
        # Transaction occurs after sender's birthday
        sender_birth = account_birthdays[sender]
        dt = sender_birth + timedelta(seconds=int(np.random.randint(0, 10 * 86400)))
        amt = round(float(np.random.exponential(scale=250.0) + 10.0), 2)
        fmt = np.random.choice(formats, p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        records.append({
            "Timestamp": dt.strftime("%Y/%m/%d %H:%M"),
            "From Bank": np.random.choice(banks),
            "Account": sender,
            "To Bank": np.random.choice(banks),
            "Account.1": receiver,
            "Amount Received": amt,
            "Receiving Currency": "USD",
            "Amount Paid": amt,
            "Payment Currency": "USD",
            "Payment Format": fmt,
            "Is Laundering": 0
        })
        
    # 2. Hard Negatives (Small businesses with high volume in/out but non-mule dynamics)
    hard_negatives = list(normal_accs)[:20]
    for b_acc in hard_negatives:
        base_time = account_birthdays[b_acc]
        for i in range(15):
            sender = np.random.choice(list(normal_accs))
            dt_in = base_time + timedelta(hours=i*4 + np.random.randint(0, 60))
            amt_in = round(float(np.random.uniform(500, 3000)), 2)
            records.append({
                "Timestamp": dt_in.strftime("%Y/%m/%d %H:%M"),
                "From Bank": np.random.choice(banks),
                "Account": sender,
                "To Bank": np.random.choice(banks),
                "Account.1": b_acc,
                "Amount Received": amt_in,
                "Receiving Currency": "USD",
                "Amount Paid": amt_in,
                "Payment Currency": "USD",
                "Payment Format": "ACH",
                "Is Laundering": 0
            })
            # Business pays out supplier after delay
            dt_out = dt_in + timedelta(days=2, hours=np.random.randint(1, 10))
            amt_out = round(amt_in * 0.75, 2) # retains margin
            supplier = np.random.choice(list(normal_accs))
            records.append({
                "Timestamp": dt_out.strftime("%Y/%m/%d %H:%M"),
                "From Bank": np.random.choice(banks),
                "Account": b_acc,
                "To Bank": np.random.choice(banks),
                "Account.1": supplier,
                "Amount Received": amt_out,
                "Receiving Currency": "USD",
                "Amount Paid": amt_out,
                "Payment Currency": "USD",
                "Payment Format": "Wire",
                "Is Laundering": 0
            })
            
    # 3. Money Mule Laundering Typologies (Fan-In -> Rapid Pass-Through -> Fan-Out)
    for mule in mule_accs:
        # Mule account starts activity at its creation date
        mule_start = account_birthdays[mule]
        
        # Criminal origin sends funds (scatter to mule)
        n_inflow = np.random.randint(2, 6)
        total_mule_in = 0.0
        
        for k in range(n_inflow):
            orig = np.random.choice(list(origin_accs))
            t_in = mule_start + timedelta(minutes=int(k * np.random.randint(10, 120)))
            # Structuring amount (e.g. 8,500 - 9,900 USD)
            amt_in = round(float(np.random.uniform(8500, 9900)), 2)
            total_mule_in += amt_in
            
            records.append({
                "Timestamp": t_in.strftime("%Y/%m/%d %H:%M"),
                "From Bank": np.random.choice(banks),
                "Account": orig,
                "To Bank": np.random.choice(banks),
                "Account.1": mule,
                "Amount Received": amt_in,
                "Receiving Currency": "USD",
                "Amount Paid": amt_in,
                "Payment Currency": "USD",
                "Payment Format": "Wire",
                "Is Laundering": 1
            })
            
        # Mule rapidly drains balance to destination accounts (rapid velocity, high pass-through)
        n_outflow = np.random.randint(2, 5)
        out_per_tx = total_mule_in * 0.98 / n_outflow # retains tiny commission ~2%
        
        for k in range(n_outflow):
            dest = np.random.choice(list(dest_accs))
            # Rapid transfer: within 15 mins to 4 hours after receipt
            t_out = mule_start + timedelta(hours=np.random.randint(1, 6), minutes=int(k * 15))
            amt_out = round(out_per_tx + np.random.uniform(-50, 50), 2)
            
            records.append({
                "Timestamp": t_out.strftime("%Y/%m/%d %H:%M"),
                "From Bank": np.random.choice(banks),
                "Account": mule,
                "To Bank": np.random.choice(banks),
                "Account.1": dest,
                "Amount Received": amt_out,
                "Receiving Currency": "USD",
                "Amount Paid": amt_out,
                "Payment Currency": "USD",
                "Payment Format": "Wire",
                "Is Laundering": 1
            })

    df = pd.DataFrame(records)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y/%m/%d %H:%M")
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_transactions(n_accounts=500, n_transactions=10000)
    print(f"Generated {len(df)} transactions.")
    print(f"Laundering transaction percentage: {df['Is Laundering'].mean()*100:.2f}%")
