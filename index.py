from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "best_churn_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Input data structure
# --------------------------------------------------

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: str
    Dependents: str

    tenure: int = Field(ge=0)

    PhoneService: str
    MultipleLines: str

    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str

    PaperlessBilling: str

    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# --------------------------------------------------
# Home / health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "message": "Customer Churn Prediction API is running",
        "model": "Logistic Regression"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert input into DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Get churn probability
    probability = float(
        model.predict_proba(data)[0][1]
    )

    # Prediction using default 0.5 threshold
    prediction = int(probability >= 0.5)

    # Risk category
    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(probability * 100, 2),
        "risk": risk
    }

# --------------------------------------------------
# Serve frontend
# --------------------------------------------------

app.mount(
    "/",
    StaticFiles(
        directory=BASE_DIR / "public",
        html=True
    ),
    name="public"
)