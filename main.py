"""
FastAPI application for real-time fraud detection.
Uses the trained XGBoost model to predict fraud probability for transactions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from schemas import TransactionSchema

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using XGBoost",
    version="1.0.0"
)

# Global variables for model and scaler (loaded once at startup)
model = None
scaler = None

@app.on_event("startup")
async def load_artifacts():
    """
    Load model and scaler when the API starts.
    This runs once, not on every request (much faster).
    """
    global model, scaler
    
    try:
        # Load the best model (experiment 3: cost-sensitive learning)
        model = joblib.load('models/experiment_3:_cost-sensitive_only.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✓ Model and scaler loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

class PredictionResponse(BaseModel):
    """Schema for API response"""
    fraud_probability: float
    is_fraud: bool
    threshold: float = 0.3

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Used by monitoring tools to verify the API is running.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionSchema):
    """
    Main prediction endpoint.
    
    Accepts a transaction (validated by Pydantic using your TransactionSchema),
    preprocesses it, runs through the model, and returns fraud probability.
    
    Args:
        transaction: Validated transaction data (30 features)
    
    Returns:
        fraud_probability: Float between 0-1 (confidence score)
        is_fraud: Boolean (true if probability > 0.3)
        threshold: The threshold used (0.3)
    """
    
    # Check if model is loaded
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Step 1: Convert Pydantic model to dictionary then to DataFrame
    # This creates a single-row DataFrame with all 30 features
    transaction_dict = transaction.model_dump()
    df = pd.DataFrame([transaction_dict])
    
    # Step 2: Scale Time and Amount features
    # The model was trained on scaled data, so we must scale predictions too
    # We only scale Time and Amount (V1-V28 are already scaled from PCA)
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    
    # IMPORTANT: Reorder columns to match training data
    # Model expects: Time, V1, V2, ..., V28, Amount
    column_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df = df[column_order]
    
    # Step 3: Get fraud probability from model
    # predict_proba returns [[prob_legitimate, prob_fraud]]
    # We want the fraud probability (second column, index 1)
    probabilities = model.predict_proba(df)
    fraud_probability = float(probabilities[0][1])
    
    # Step 4: Apply business rule threshold
    # Flag as fraud if probability > 30%
    # (Lower threshold = catch more fraud, but more false alarms)
    threshold = 0.3
    is_fraud = fraud_probability > threshold
    
    # Step 5: Return prediction
    return PredictionResponse(
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        threshold=threshold
    )

@app.get("/")
def root():
    """
    Root endpoint with API info.
    """
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }

