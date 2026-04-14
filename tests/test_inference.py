import pandas as pd
import torch
import joblib
from src.config import MODEL_DIR
from src.models.mlp import ChurnMLP

def predict_new_customer(customer_dict):
    # 1. Carregar o Encoder e o Modelo
    encoder = joblib.load(MODEL_DIR / "one_hot_encoder.joblib")
    
    # Precisamos saber a dimensão de entrada (número de colunas após o encoder)
    # Aqui um truque: o encoder sabe quantas colunas ele gera + as numéricas
    # Para este teste, vamos carregar o modelo salvo pelo MLflow ou o arquivo local
    
    # 2. Transformar o dicionário em DataFrame
    df_new = pd.DataFrame([customer_dict])

    # 3. Pré-processamento manual (o que o clean_data fazia)
    if 'customerID' in df_new.columns:
        df_new = df_new.drop(columns=['customerID'])
    df_new['TotalCharges'] = pd.to_numeric(df_new['TotalCharges'], errors='coerce').fillna(0)

    # 4. Aplicar o OneHotEncoder salvo
    cat_cols = df_new.select_dtypes(include=['object']).columns.tolist()
    num_cols = df_new.select_dtypes(exclude=['object']).columns.tolist()
    
    encoded_data = encoder.transform(df_new[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
    
    X_final = pd.concat([df_new[num_cols], encoded_df], axis=1)

    # 5. Converter para Tensor
    X_tensor = torch.tensor(X_final.values, dtype=torch.float32)

    # 6. Carregar modelo e prever
    input_dim = X_tensor.shape[1]
    model = ChurnMLP(input_dim)
    # Se você salvou o state_dict, carregue-o aqui. 
    # Por enquanto, vamos ver se a transformação de dados passa:
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        probability = torch.sigmoid(output).item()
    
    return probability

# --- TESTE COM UM CLIENTE NOVO ---
new_client = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
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