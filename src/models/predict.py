import torch
import pandas as pd
import joblib
from src.models.mlp import ChurnMLP
from src.features.build_features import transform_features
from src.config import MODEL_DIR, MODEL_PATH

def load_model(input_dim):
    model = ChurnMLP(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def predict(df: pd.DataFrame):
    # 1. Transforma as features (gera dummies/scaler)
    X = transform_features(df)

    # 2. Carrega a ordem correta das colunas
    try:
        expected_columns = joblib.load(MODEL_DIR / "feature_names.joblib")
        # Reordena o dataframe e preenche com 0 colunas que possam faltar
        X = X.reindex(columns=expected_columns, fill_value=0)
    except FileNotFoundError:
        print("Aviso: feature_names.joblib não encontrado. Usando ordem atual.")

    # 3. Converte para Tensor 
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    
    # 4. Carrega modelo e faz a predição
    model = load_model(X_tensor.shape[1])

    with torch.no_grad():
        # Aplica sigmoid para obter probabilidade entre 0 e 1
        probs = torch.sigmoid(model(X_tensor)).numpy()

    return probs



def predict_new_customer(customer_dict: dict) -> float:
    """
    Função wrapper para ser usada pela API FastAPI.
    Recebe um dicionário com os dados de um cliente, converte para DataFrame,
    faz a predição e retorna a probabilidade como um float puro.
    """
    # 1. Converte o dicionário que veio da API em um DataFrame de 1 linha
    df_new = pd.DataFrame([customer_dict])
    
    # 2. Como TotalCharges pode vir como string da API (igual ao dataset bruto), 
    # garantimos que vire número antes de passar pelo transform_features
    if 'TotalCharges' in df_new.columns:
        df_new['TotalCharges'] = pd.to_numeric(df_new['TotalCharges'], errors='coerce').fillna(0)

    # 3. Chama a sua função predict que já está pronta e funciona super bem!
    probs_array = predict(df_new)
    
    # 4. probs_array é um array 2D do numpy, por exemplo: [[0.745]]
    # Precisamos extrair esse valor e converter para o tipo float do Python
    probability = float(probs_array[0][0])
    
    return probability