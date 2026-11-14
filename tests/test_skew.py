"""Test de training-serving skew: datos y predicciones consistentes."""
import os
import requests
import pytest


API_URL = os.getenv("API_URL", "http://localhost:8000")


@pytest.mark.parametrize("customer", [
    {
        "gender": "Male",
        "senior_citizen": 0,
        "tenure_months": 12,
        "contract_type": "Month-to-Month",
        "internet_service": "Fiber",
        "monthly_charges": 70.35,
        "support_tickets_last_6m": 3,
    },
    {
        "gender": "Female",
        "senior_citizen": 1,
        "tenure_months": 60,
        "contract_type": "Two Year",
        "internet_service": "DSL",
        "monthly_charges": 85.5,
        "support_tickets_last_6m": 0,
    },
])
def test_api_returns_valid_prediction(customer):
    """Validar que el endpoint /predict retorna predicciones válidas."""
    res = requests.post(f"{API_URL}/predict", json=customer)
    assert res.status_code == 200

    data = res.json()
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_level"] in ["BAJO", "MEDIO", "ALTO"]


def test_consistent_predictions():
    """Mismos inputs → mismas predicciones (no hay skew)."""
    customer = {
        "gender": "Male",
        "senior_citizen": 0,
        "tenure_months": 12,
        "contract_type": "Month-to-Month",
        "internet_service": "Fiber",
        "monthly_charges": 70.35,
        "support_tickets_last_6m": 3,
    }

    res1 = requests.post(f"{API_URL}/predict", json=customer)
    res2 = requests.post(f"{API_URL}/predict", json=customer)

    assert res1.json()["churn_probability"] == res2.json()["churn_probability"]


def test_model_info_accessible():
    """Verificar que /model/info retorna metadata del modelo."""
    res = requests.get(f"{API_URL}/model/info")
    assert res.status_code == 200

    data = res.json()
    assert "version" in data
    assert "model_type" in data
    assert "threshold" in data
