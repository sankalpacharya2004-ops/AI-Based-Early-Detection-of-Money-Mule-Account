import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import MuleDetectorModels
from src.explainability import MuleSHAPExplainer

app = FastAPI(
    title="Money Mule Early Detection API",
    description="Real-time early risk scoring and SHAP explainability API for financial account monitoring",
    version="1.0.0"
)

# Global variables for model state
model_engine = None
xgb_model = None
shap_explainer = None
feature_columns = []

class AccountFeatureInput(BaseModel):
    features: Dict[str, float]

@app.on_event("startup")
def load_artifacts():
    global model_engine, xgb_model, shap_explainer, feature_columns
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mule_models.pkl")
    feats_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "account_features_7d.csv")
    
    if os.path.exists(model_path):
        model_engine = MuleDetectorModels()
        model_engine.load_models(model_path)
        xgb_model = model_engine.models.get("XGBoost")
        
        if os.path.exists(feats_path):
            sample_df = pd.read_csv(feats_path)
            feature_columns = [c for c in sample_df.columns if c != "account_id"]
            if xgb_model:
                shap_explainer = MuleSHAPExplainer(xgb_model, sample_df[feature_columns].head(50))
    else:
        print("Warning: Model artifacts not found. Please run 'python run_pipeline.py' first.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": xgb_model is not None,
        "features_count": len(feature_columns)
    }

@app.post("/score")
def score_account(payload: AccountFeatureInput):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run training pipeline first.")
        
    feats_dict = payload.features
    df_input = pd.DataFrame([feats_dict])
    
    # Fill missing features with 0.0
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0.0
            
    df_input = df_input[feature_columns]
    
    risk_prob = float(xgb_model.predict_proba(df_input)[0, 1])
    is_high_risk = risk_prob > 0.65
    
    return {
        "mule_risk_score": round(risk_prob, 4),
        "is_flagged_mule": is_high_risk,
        "risk_level": "CRITICAL" if risk_prob > 0.8 else ("HIGH" if risk_prob > 0.65 else ("MEDIUM" if risk_prob > 0.4 else "LOW"))
    }

@app.post("/explain")
def explain_account(payload: AccountFeatureInput):
    if shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not loaded.")
        
    feats_dict = payload.features
    df_input = pd.DataFrame([feats_dict])
    
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0.0
            
    df_input = df_input[feature_columns]
    
    shap_df = shap_explainer.explain_account(df_input)
    explanation = shap_df.to_dict(orient="records")
    
    return {
        "account_explanation": explanation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
