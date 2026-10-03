import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from typing import Dict, Any

class MuleDetectorModels:
    def __init__(self):
        self.models = {}
        
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Trains Logistic Regression, Random Forest, XGBoost, and Isolation Forest models.
        Applies imbalance handling via class weighting / scale_pos_weight.
        """
        # Calculate imbalance scale weight for XGBoost
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = max(neg_count / max(pos_count, 1), 1.0)
        
        # 1. Baseline: Logistic Regression
        logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        logreg.fit(X_train, y_train)
        self.models["LogisticRegression"] = logreg
        
        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            class_weight="balanced_subsample", 
            n_jobs=-1, 
            random_state=42
        )
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        
        # 3. XGBoost (Main Model)
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb
        
        # 4. Unsupervised Anomaly Detection: Isolation Forest
        iso = IsolationForest(
            n_estimators=150, 
            contamination=0.05, 
            random_state=42
        )
        iso.fit(X_train)
        self.models["IsolationForest"] = iso
        
        return self.models

    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
            
        if model_name == "IsolationForest":
            # Convert decision function to a 0-1 risk score (lower decision_function = higher anomaly)
            scores = -model.decision_function(X)
            # Min-max scale to 0-1 range
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            return scores_norm
        else:
            return model.predict_proba(X)[:, 1]

    def save_models(self, path: str = "models/mule_models.pkl"):
        joblib.dump(self.models, path)

    def load_models(self, path: str = "models/mule_models.pkl"):
        self.models = joblib.load(path)
