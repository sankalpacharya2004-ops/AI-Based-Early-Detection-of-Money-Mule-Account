import pandas as pd
import numpy as np
import shap

class MuleSHAPExplainer:
    def __init__(self, model, X_sample: pd.DataFrame):
        """
        Initializes SHAP TreeExplainer for XGBoost / RandomForest models.
        """
        self.model = model
        self.feature_names = list(X_sample.columns)
        self.explainer = shap.TreeExplainer(model)
        
    def get_global_importance(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mean absolute SHAP values for each feature across the test dataset.
        """
        shap_values = self.explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            # For multi-output models or binary RF
            shap_values = shap_values[1]
            
        mean_shap = np.abs(shap_values).mean(axis=0)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "mean_shap_importance": mean_shap
        }).sort_values("mean_shap_importance", ascending=False).reset_index(drop=True)
        return df_imp

    def explain_account(self, account_features: pd.DataFrame) -> pd.DataFrame:
        """
        Explains a single account's mule risk prediction by detailing each feature's contribution.
        """
        shap_values = self.explainer.shap_values(account_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        row = account_features.iloc[0]
        s_vals = shap_values[0]
        
        df_local = pd.DataFrame({
            "feature": self.feature_names,
            "feature_value": row.values,
            "shap_contribution": s_vals
        }).sort_values("shap_contribution", ascending=False).reset_index(drop=True)
        
        return df_local
