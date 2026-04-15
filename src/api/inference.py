import pandas as pd
import torch
import joblib
from src.models.mlp import ChurnMLP
from src.config import MODEL_DIR

class ChurnPredictor:
    """
    Classe unificada para inferência. 
    Carrega os artefatos de dados e o modelo PyTorch.
    """
    def __init__(self, input_dim: int = 30):
        # 1. Carregar os transformadores do Scikit-Learn
        self.encoder = joblib.load(MODEL_DIR / 'one_hot_encoder.joblib')
        self.scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
        
        # 2. Carregar o modelo PyTorch
        self.model = ChurnMLP(input_dim=input_dim)
        model_path = MODEL_DIR / 'mlp_churn_best.pt'
        self.model.load_state_dict(torch.load(model_path))
        
        # 3. Colocar o modelo em modo de avaliação (desativa Dropout)
        self.model.eval()

    def preprocess_input(self, customer_data: dict) -> torch.Tensor:
        """Transforma o dicionário recebido da API no tensor que a rede espera."""
        df = pd.DataFrame([customer_data])
        
        # Tratar TotalCharges e CustomerID
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            
        # Separar colunas para transformação
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
        
        # Aplicar Encoder e Scaler
        cat_encoded = self.encoder.transform(df[categorical_cols])
        num_scaled = self.scaler.transform(df[numerical_cols])
        
        # Recuperar nomes e juntar DataFrames
        cat_cols_names = self.encoder.get_feature_names_out(categorical_cols)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_cols_names)
        num_df = pd.DataFrame(num_scaled, columns=numerical_cols)
        
        X_final = pd.concat([num_df, cat_df], axis=1)
        
        # Converter para tensor float32
        return torch.tensor(X_final.values, dtype=torch.float32)

    def predict(self, customer_data: dict, threshold: float = 0.5) -> dict:
        """Gera a predição final para a API."""
        # Processar os dados
        input_tensor = self.preprocess_input(customer_data)
        
        # Fazer a predição sem calcular gradientes (economiza memória)
        with torch.no_grad():
            # A rede retorna os logits puros (pois removemos a Sigmoid antes)
            logits = self.model(input_tensor)
            # Aplicar Sigmoid para obter a probabilidade entre 0 e 1
            probability = torch.sigmoid(logits).item()
            
        # Definir a classe com base no threshold (útil para o tradeoff precision/recall do EDA)
        churn_prediction = 1 if probability >= threshold else 0
        
        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": churn_prediction,
            "threshold_used": threshold
        }