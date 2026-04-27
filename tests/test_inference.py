import joblib
import pandas as pd
import pytest
import torch

from src.config import MODEL_DIR
from src.models.mlp import ChurnMLP


@pytest.fixture
def sample_customer():
    return {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
        'tenure': 15, 'PhoneService': 'Yes', 'MultipleLines': 'No phone service',
        'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'No', 'TechSupport': 'Yes', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 70.0, 'TotalCharges': '1700'
    }

def test_mlp_inference_smoke(sample_customer):
    """
    Smoke test: Garante que o pipeline de inferência (encoder + scaler + MLP)
    funciona do início ao fim sem erros.
    """
    # 1. Carregar artefatos
    encoder = joblib.load(MODEL_DIR / "one_hot_encoder.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    # 2. Processar input
    df_new = pd.DataFrame([sample_customer])
    df_new['TotalCharges'] = pd.to_numeric(df_new['TotalCharges'], errors='coerce').fillna(0)

    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    encoded_data = encoder.transform(df_new[cat_cols])
    scaled_data = scaler.transform(df_new[num_cols])

    X_final = pd.concat([
        pd.DataFrame(scaled_data, columns=num_cols),
        pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
    ], axis=1)

    # 3. Carregar Modelo
    X_tensor = torch.tensor(X_final.values, dtype=torch.float32)
    model = ChurnMLP(X_tensor.shape[1])
    model.load_state_dict(torch.load(MODEL_DIR / "mlp_churn_best.pt"))
    model.eval()

    # 4. Predição
    with torch.no_grad():
        output = model(X_tensor)
        probability = torch.sigmoid(output).item()

    # Asserts
    assert 0.0 <= probability <= 1.0, "A probabilidade deve estar entre 0 e 1"
    assert isinstance(probability, float)
