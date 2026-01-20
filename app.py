import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ===============================
# CARGA DEL MODELO (PIPELINE PURO)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_xgboost_churn.pkl")

pipeline = joblib.load(MODEL_PATH)

# ⚠️ Threshold fijo (porque no viene en el PKL)
THRESHOLD = 0.5

# Columnas esperadas (DEBEN coincidir con el entrenamiento)
COLUMNAS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "contract",
    "payment_method",
    "internet_service",
    "tech_support",
    "online_security"
]

# ===============================
# FASTAPI
# ===============================
app = FastAPI(title="API Predicción de Churn")

class Cliente(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    support_calls: int
    contract: str
    payment_method: str
    internet_service: str
    tech_support: str
    online_security: str


@app.post("/predict")
async def predecir_churn(cliente: Cliente):
    try:
        df = pd.DataFrame([cliente.dict()])
        df = df[COLUMNAS]

        prob = pipeline.predict_proba(df)[:, 1][0]
        churn = int(prob >= THRESHOLD)

        return {
            "Churn": bool(churn),
            "ProbabilidadChurn": round(float(prob), 4),
            "Threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

