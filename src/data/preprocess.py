import pandas as pd
from src.middleware.logger import get_logger


logger = get_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial:
    - Converte TotalCharges para numérico
    - Trata valores nulos
    - Remove CustomerID
    """
    df = df.copy()

    # 1. Converter TotalCharges (que vem como string/object) para float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 2. Tratar valores nulos
    null_count = df['TotalCharges'].isnull().sum()
    if null_count > 0:
        logger.info(f"Preenchendo {null_count} valores nulos em TotalCharges com 0")
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Remover coluna customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 4. Codificar variável alvo (Churn) para numérico
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    logger.info("Limpeza de dados concluída.")
    return df