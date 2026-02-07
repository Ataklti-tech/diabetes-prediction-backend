import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from backend.main import app

sys.path.append(str(Path(__file__).resolve().parent.parent))

client = TestClient(app)



def test_prediction_success():
    payload = {
        "pregnancies": 12,
        "glucose": 200,
        "blood_pressure": 90,
        "skin_thickness": 2,
        "insulin": 0,
        "bmi": 29.6,
        "diabetes_pedigree_function": 0.351,
        "age": 53
}
    response = client.post("/predict", json=payload)
    
    # Print the full response JSON
    print("\n=== PREDICTION RESULT ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    print("=========================\n")

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [0, 1]


def test_prediction_invalid_bmi():
    payload = {
        "pregnancies": 2,
        "glucose": 130.0,
        "blood_pressure": 80.0,
        "skin_thickness": 20.0,
        "insulin": 90.0,
        "bmi": 28.5,
        "diabetes_pedigree_function": 0.5,
        "age": 45
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422  

def test_prediction_missing_field():
    payload = {
        "age": 45,
        "bmi": 25.5,
        "glucose": 130
        # missing fields
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
