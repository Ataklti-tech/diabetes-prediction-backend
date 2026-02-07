import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Dict, Any
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Professional API for predicting diabetes risk using machine learning by Ataklti Okbe",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# Initialize model and scaler
model = None
scaler = None

try:
    # Load model
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
    
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler loaded successfully")
    
    # Log scaler information
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        logger.info(f"Scaler means: {scaler.mean_}")
        logger.info(f"Scaler scales: {scaler.scale_}")
    
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except Exception as e:
    logger.error(f"Error loading model/scaler: {str(e)}")

# Input schema
class PredictionInput(BaseModel):
    """Input schema for diabetes prediction"""
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=300, description="Glucose level (mg/dL)")
    blood_pressure: float = Field(..., ge=0, le=200, description="Blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Skin thickness (mm)")
    insulin: float = Field(..., ge=0, le=900, description="Insulin level (μU/mL)")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index")
    diabetes_pedigree_function: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., ge=1, le=120, description="Age in years")

    @field_validator("bmi")
    @classmethod
    def validate_bmi(cls, v: float) -> float:
        if not 10 <= v <= 70:
            raise ValueError("BMI must be between 10 and 70")
        return v

    @field_validator("glucose")
    @classmethod
    def validate_glucose(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Glucose level cannot be negative")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pregnancies": 2,
                "glucose": 120.0,
                "blood_pressure": 70.0,
                "skin_thickness": 20.0,
                "insulin": 80.0,
                "bmi": 25.5,
                "diabetes_pedigree_function": 0.5,
                "age": 33
            }
        }
    )

# Output schema
class PredictionOutput(BaseModel):
    """Output schema for diabetes prediction"""
    prediction: int = Field(..., description="Prediction result (0: No diabetes, 1: Diabetes)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    confidence: Optional[float] = Field(None, description="Confidence score if available")
    timestamp: str = Field(..., description="Prediction timestamp")
    input_data: Dict[str, Any] = Field(..., description="Input parameters used for prediction")
    recommendation: str = Field(..., description="Health recommendation based on prediction")

# Health status schema
class HealthStatus(BaseModel):
    """API health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    scaler_loaded: bool
    version: str

# Helper functions
def get_risk_level(prediction: int, input_data: dict) -> str:
    """Determine risk level based on prediction and input parameters"""
    if prediction == 0:
        # Check for risk factors even if prediction is negative
        risk_factors = 0
        if input_data['glucose'] > 140:
            risk_factors += 1
        if input_data['bmi'] > 30:
            risk_factors += 1
        if input_data['age'] > 45:
            risk_factors += 1
        if input_data['blood_pressure'] > 80:
            risk_factors += 1
        
        if risk_factors >= 2:
            return "Medium"
        else:
            return "Low"
    else:
        # For diabetes prediction, determine severity
        if input_data['glucose'] > 200 or input_data['age'] > 60:
            return "High"
        else:
            return "Medium"

def get_recommendation(prediction: int, risk_level: str) -> str:
    """Generate health recommendation based on prediction"""
    recommendations = {
        "Low": "Maintain a healthy lifestyle with regular exercise and balanced diet. Continue regular health check-ups.",
        "Medium": "Consider lifestyle modifications including regular exercise, healthy diet, and stress management. Schedule a consultation with your healthcare provider.",
        "High": "It is strongly recommended to consult with a healthcare professional immediately for comprehensive evaluation and management plan."
    }
    return recommendations.get(risk_level, "Please consult with a healthcare professional for personalized advice.")

# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    return HealthStatus(
        status="healthy" if model is not None and scaler is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict_diabetes(input_data: PredictionInput):
    """
    Predict diabetes risk based on patient parameters
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or scaler not loaded. Please contact administrator."
        )
    
    try:
        # Prepare input features in correct order
        raw_features = np.array([[
            input_data.pregnancies,
            input_data.glucose,
            input_data.blood_pressure,
            input_data.skin_thickness,
            input_data.insulin,
            input_data.bmi,
            input_data.diabetes_pedigree_function,
            input_data.age
        ]])
        
        # Log raw features for debugging
        logger.info(f"Raw features: {raw_features[0]}")
        
        # Scale features using the loaded scaler
        scaled_features = scaler.transform(raw_features)
        logger.info(f"Scaled features: {scaled_features[0]}")
        
        # Make prediction
        prediction = int(model.predict(scaled_features)[0])
        logger.info(f"🎯 Prediction: {prediction}")
        
        # Get prediction probabilities if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_features)[0]
            confidence = float(max(proba))
            logger.info(f"Prediction probabilities: {proba}")
            logger.info(f"Confidence: {confidence:.2%}")
        
        # Convert input data to dict
        input_dict = input_data.model_dump()
        
        # Determine risk level
        risk_level = get_risk_level(prediction, input_dict)
        
        # Generate recommendation
        recommendation = get_recommendation(prediction, risk_level)
        
        # Log final result
        logger.info(f"Final prediction: {prediction}, Risk: {risk_level}")
        
        return PredictionOutput(
            prediction=prediction,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            input_data=input_dict,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )

@app.get("/model-info", response_model=Dict[str, Any])
async def model_info():
    """Get information about the loaded model and scaler"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or scaler not loaded"
        )
    
    info = {
        "model_type": type(model).__name__,
        "scaler_type": type(scaler).__name__,
        "features": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],
        "model_loaded": True,
        "scaler_loaded": True,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add scaler statistics if available
    if hasattr(scaler, 'mean_'):
        info["feature_means"] = scaler.mean_.tolist()
    if hasattr(scaler, 'scale_'):
        info["feature_scales"] = scaler.scale_.tolist()
    
    return info

@app.post("/test-scaling")
async def test_scaling():
    """Test endpoint to verify scaling is working correctly"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or scaler not loaded"
        )
    
    # Test cases
    test_cases = [
        {
            "name": "Non-diabetes case",
            "data": {
                "pregnancies": 1,
                "glucose": 85.0,
                "blood_pressure": 66.0,
                "skin_thickness": 29.0,
                "insulin": 0.0,
                "bmi": 26.6,
                "diabetes_pedigree_function": 0.351,
                "age": 31
            }
        },
        {
            "name": "Diabetes case",
            "data": {
                "pregnancies": 6,
                "glucose": 148.0,
                "blood_pressure": 72.0,
                "skin_thickness": 35.0,
                "insulin": 0.0,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50
            }
        }
    ]
    
    results = []
    
    for test in test_cases:
        raw_features = np.array([list(test["data"].values())])
        scaled_features = scaler.transform(raw_features)
        
        prediction = int(model.predict(scaled_features)[0])
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_features)[0]
            confidence = float(max(proba))
        else:
            confidence = None
        
        results.append({
            "test_name": test["name"],
            "raw_features": raw_features[0].tolist(),
            "scaled_features": scaled_features[0].tolist(),
            "prediction": prediction,
            "confidence": confidence,
            "interpretation": "No Diabetes" if prediction == 0 else "Diabetes Detected"
        })
    
    return {
        "message": "Scaling test results",
        "scaler_type": type(scaler).__name__,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)