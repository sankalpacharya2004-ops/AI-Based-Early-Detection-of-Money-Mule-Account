import pandas as pd

def derive_account_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives defensible account-level proxy ground truth labels from transaction data:
    - 1 = Money Mule (Pass-through account: receives AND sends illicit/laundering transactions)
    - 0 = Normal / Non-mule account (includes origin senders, final receivers, and legitimate accounts)
    
    Returns a DataFrame with columns: ['account_id', 'is_mule', 'account_role']
    """
    # Identify unique accounts
    all_accounts = pd.unique(df[['Account', 'Account.1']].values.ravel())
    
    # Filter illicit transactions
    illicit_df = df[df['Is Laundering'] == 1]
    
    illicit_receivers = set(illicit_df['Account.1'].unique())
    illicit_senders = set(illicit_df['Account'].unique())
    
    # Pass-through accounts (received dirty money AND forwarded dirty money)
    mules = illicit_receivers.intersection(illicit_senders)
    origins = illicit_senders - mules
    destinations = illicit_receivers - mules
    
    labels = []
    for acc in all_accounts:
        if acc in mules:
            role = "mule"
            is_mule = 1
        elif acc in origins:
            role = "origin"
            is_mule = 0
        elif acc in destinations:
            role = "destination"
            is_mule = 0
        else:
            role = "normal"
            is_mule = 0
            
        labels.append({
            "account_id": acc,
            "is_mule": is_mule,
            "account_role": role
        })
        
    label_df = pd.DataFrame(labels)
    return label_df

if __name__ == "__main__":
    import os
    from generator import generate_synthetic_transactions
    df = generate_synthetic_transactions()
    labels = derive_account_labels(df)
    print(labels['account_role'].value_counts())
