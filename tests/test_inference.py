import pandas as pd
import torch
import joblib
from src.config import MODEL_DIR
from src.models.mlp import ChurnMLP

def predict_new_customer(customer_dict):
    # 1. Carregar os Transformadores (Encoder e Scaler)
    encoder = joblib.load(MODEL_DIR / "one_hot_encoder.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib") # Adicionamos o scaler!
    
    # 2. Transformar o dicionário em DataFrame e Limpar
    df_new = pd.DataFrame([customer_dict])
    if 'customerID' in df_new.columns:
        df_new = df_new.drop(columns=['customerID'])
    df_new['TotalCharges'] = pd.to_numeric(df_new['TotalCharges'], errors='coerce').fillna(0)

    # 3. Separar colunas para aplicar as transformações
    cat_cols = df_new.select_dtypes(include=['object']).columns.tolist()
    num_cols = df_new.select_dtypes(exclude=['object']).columns.tolist()
    
    # Aplicar o Encoder e o Scaler que foram salvos no treino
    encoded_data = encoder.transform(df_new[cat_cols])
    scaled_data = scaler.transform(df_new[num_cols]) # Importante: Normalizar os números!
    
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
    scaled_df = pd.DataFrame(scaled_data, columns=num_cols)
    
    X_final = pd.concat([scaled_df, encoded_df], axis=1)

    # 4. Converter para Tensor
    X_tensor = torch.tensor(X_final.values, dtype=torch.float32)

    # 5. CARREGAR O MODELO TREINADO (A parte que faltava)
    input_dim = X_tensor.shape[1]
    model = ChurnMLP(input_dim)
    
    # Carrega os pesos do arquivo gerado no train.py
    model_path = MODEL_DIR / "mlp_churn_best.pt"
    model.load_state_dict(torch.load(model_path))
    
    model.eval() # Modo de avaliação
    with torch.no_grad():
        output = model(X_tensor)
        # Como no mlp.py tiramos a Sigmoid da arquitetura, aplicamos ela aqui
        probability = torch.sigmoid(output).item()
    
    return probability

# --- TESTE COM UM CLIENTE NOVO ---
new_client = {
    'gender': 'Female',
    'SeniorCitizen': 1,
    'Partner': 'Yes',
    'Dependents': 'Yes',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': '29.85'
}

prob = predict_new_customer(new_client)
print(f"Probabilidade de Churn: {prob:.2%}")
print("Resultado: Provável Churn" if prob > 0.5 else "Resultado: Fiel")