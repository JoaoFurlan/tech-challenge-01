from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}

def test_predict_endpoint():
    # Os mesmos dados de teste do seu test_inference.py
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 15,
        "PhoneService": "Yes",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": "1700"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    dados = response.json()
    assert "churn_probability" in dados
    assert "churn_prediction" in dados
    assert "message" in dados
    assert isinstance(dados["churn_probability"], float)
