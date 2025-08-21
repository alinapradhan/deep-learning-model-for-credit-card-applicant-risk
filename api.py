"""
FastAPI application for real-time credit card fraud detection.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import uvicorn
import os
from datetime import datetime
import logging
import shap

from model import CreditRiskModel
from data_preprocessing import DataPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time credit risk assessment for credit card applicants",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
preprocessor = None
explainer = None

class CreditCardApplication(BaseModel):
    """Input schema for credit card application."""
    limit_bal: float = Field(..., description="Credit limit balance", ge=0)
    sex: int = Field(..., description="Gender (1=male, 2=female)", ge=1, le=2)
    education: int = Field(..., description="Education level (1-4)", ge=1, le=4)
    marriage: int = Field(..., description="Marriage status (1-3)", ge=1, le=3)
    age: int = Field(..., description="Age in years", ge=18, le=100)
    pay_0: int = Field(..., description="Repayment status in September", ge=-2, le=9)
    pay_2: int = Field(..., description="Repayment status in August", ge=-2, le=9)
    pay_3: int = Field(..., description="Repayment status in July", ge=-2, le=9)
    pay_4: int = Field(..., description="Repayment status in June", ge=-2, le=9)
    pay_5: int = Field(..., description="Repayment status in May", ge=-2, le=9)
    pay_6: int = Field(..., description="Repayment status in April", ge=-2, le=9)
    bill_amt1: float = Field(..., description="Bill statement amount September")
    bill_amt2: float = Field(..., description="Bill statement amount August")
    bill_amt3: float = Field(..., description="Bill statement amount July")
    bill_amt4: float = Field(..., description="Bill statement amount June")
    bill_amt5: float = Field(..., description="Bill statement amount May")
    bill_amt6: float = Field(..., description="Bill statement amount April")
    pay_amt1: float = Field(..., description="Payment amount September", ge=0)
    pay_amt2: float = Field(..., description="Payment amount August", ge=0)
    pay_amt3: float = Field(..., description="Payment amount July", ge=0)
    pay_amt4: float = Field(..., description="Payment amount June", ge=0)
    pay_amt5: float = Field(..., description="Payment amount May", ge=0)
    pay_amt6: float = Field(..., description="Payment amount April", ge=0)

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_category: str = Field(..., description="Risk category (Low/Medium/High)")
    recommendation: str = Field(..., description="Approval recommendation")
    confidence_score: float = Field(..., description="Model confidence (0-1)")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Feature importance explanation")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    version: str

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup."""
    global model, preprocessor, explainer
    
    try:
        # Load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor('models')
        logger.info("Preprocessor loaded successfully")
        
        # Load model
        model = CreditRiskModel()
        model.load_model('models/credit_risk_model.h5')
        logger.info("Model loaded successfully")
        
        # Initialize SHAP explainer for feature importance
        # We'll use a simple explainer for now
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise e

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_default_risk(application: CreditCardApplication):
    """
    Predict default risk for a credit card application.
    
    Returns the probability of default, risk category, and recommendation.
    """
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([application.dict()])
        
        # Rename columns to match expected format
        column_mapping = {
            'limit_bal': 'LIMIT_BAL',
            'sex': 'SEX',
            'education': 'EDUCATION',
            'marriage': 'MARRIAGE',
            'age': 'AGE',
            'pay_0': 'PAY_0',
            'pay_2': 'PAY_2',
            'pay_3': 'PAY_3',
            'pay_4': 'PAY_4',
            'pay_5': 'PAY_5',
            'pay_6': 'PAY_6',
            'bill_amt1': 'BILL_AMT1',
            'bill_amt2': 'BILL_AMT2',
            'bill_amt3': 'BILL_AMT3',
            'bill_amt4': 'BILL_AMT4',
            'bill_amt5': 'BILL_AMT5',
            'bill_amt6': 'BILL_AMT6',
            'pay_amt1': 'PAY_AMT1',
            'pay_amt2': 'PAY_AMT2',
            'pay_amt3': 'PAY_AMT3',
            'pay_amt4': 'PAY_AMT4',
            'pay_amt5': 'PAY_AMT5',
            'pay_amt6': 'PAY_AMT6'
        }
        
        input_data = input_data.rename(columns=column_mapping)
        
        # Ensure all expected features are present
        expected_features = preprocessor.feature_names
        for feature in expected_features:
            if feature not in input_data.columns:
                input_data[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        input_data = input_data[expected_features]
        
        # Preprocess the data
        input_scaled = preprocessor.transform(input_data)
        
        # Make prediction
        default_probability = float(model.predict(input_scaled)[0])
        
        # Determine risk category
        if default_probability < 0.3:
            risk_category = "Low"
            recommendation = "Approve"
        elif default_probability < 0.7:
            risk_category = "Medium"
            recommendation = "Review Required"
        else:
            risk_category = "High"
            recommendation = "Reject"
        
        # Calculate confidence score (based on distance from 0.5)
        confidence_score = float(abs(default_probability - 0.5) * 2)
        
        # Simple feature importance explanation
        feature_values = input_scaled.iloc[0].values
        feature_names = input_scaled.columns.tolist()
        
        # Get top contributing features (simplified)
        feature_importance = model.get_feature_importance_scores(
            input_scaled.values, feature_names
        )
        
        top_features = feature_importance.head(5).to_dict('records')
        
        explanation = {
            "top_contributing_features": top_features,
            "input_summary": {
                "age": application.age,
                "credit_limit": application.limit_bal,
                "education_level": application.education,
                "recent_payment_status": application.pay_0
            }
        }
        
        return PredictionResponse(
            default_probability=default_probability,
            risk_category=risk_category,
            recommendation=recommendation,
            confidence_score=confidence_score,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(applications: List[CreditCardApplication]):
    """
    Predict default risk for multiple credit card applications.
    """
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for app in applications:
            # Reuse the single prediction logic
            result = await predict_default_risk(app)
            results.append(result)
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "model_type": "Deep Neural Network",
            "input_features": len(preprocessor.feature_names) if preprocessor else 0,
            "feature_names": preprocessor.feature_names if preprocessor else [],
            "model_architecture": "Sequential DNN with 4 hidden layers",
            "training_framework": "TensorFlow/Keras",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Card Fraud Detection API",
        "description": "Real-time credit risk assessment for banking applications",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - Single prediction",
            "batch_predict": "/predict/batch - Batch predictions",
            "health": "/health - Health check",
            "model_info": "/model/info - Model information",
            "docs": "/docs - API documentation"
        }
    }

def run_server(host="0.0.0.0", port=8000, reload=False):
    """Run the FastAPI server."""
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server(reload=True)