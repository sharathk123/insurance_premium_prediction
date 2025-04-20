from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from src.preprocess import preprocess_data

# App init
app = FastAPI(title="🏥 Insurance Premium Predictor", version="1.0")

# Model paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "insurance_model.pkl"
FEATURE_PATH = BASE_DIR / "model" / "feature_columns.pkl"

# Load model and feature columns
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# Pydantic model for input
class InsuranceInput(BaseModel):
    age: int
    gender: str
    bmi: float
    children: int
    smoker: str
    region: str
    medical_history: str
    family_medical_history: str
    exercise_frequency: str
    occupation: str
    coverage_level: str


@app.get("/")
def read_root():
    return {"message": "🏥 Welcome to the Insurance Premium Prediction API!"}


@app.post("/predict")
def predict(data: InsuranceInput):
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # Preprocess
    processed_df = preprocess_data(df)

    # Align features
    for col in feature_columns:
        if col not in processed_df:
            processed_df[col] = 0
    processed_df = processed_df[feature_columns]

    # Prediction
    prediction = model.predict(processed_df)[0]
    return {"predicted_premium": round(prediction, 2)}
