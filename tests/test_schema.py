import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.data.load_data import load_data


def test_load_data_schema_success(tmp_path):
    """Garante que dados válidos passam pela validação."""
    data = {
        "customerID": ["7590-VHVEG"], "gender": ["Male"], "SeniorCitizen": [0],
        "Partner": ["Yes"], "Dependents": ["No"], "tenure": [1],
        "PhoneService": ["Yes"], "MultipleLines": ["No"], "InternetService": ["DSL"],
        "OnlineSecurity": ["No"], "OnlineBackup": ["No"], "DeviceProtection": ["No"],
        "TechSupport": ["No"], "StreamingTV": ["No"], "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"], "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"], "MonthlyCharges": [10.0],
        "TotalCharges": ["10.0"], "Churn": ["No"]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "valid.csv"
    df.to_csv(file_path, index=False)

    df_loaded = load_data(str(file_path))
    assert len(df_loaded) == 1

def test_load_data_schema_failure(tmp_path):
    """Garante que dados com valores inválidos (ex: gender errado) disparam erro."""
    data = {"customerID": ["1"], "gender": ["INVALID_GENDER"]} # Erro aqui
    df = pd.DataFrame(data)
    file_path = tmp_path / "invalid.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(SchemaError):
        load_data(str(file_path))
