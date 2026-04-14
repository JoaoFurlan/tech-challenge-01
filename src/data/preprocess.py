import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.config import RANDOM_STATE

logger = get_logger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset bruto."""
    logger.info(f"Carregando dados de: {path}")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial:
    - Converte TotalCharges para numérico
    - Trata valores nulos
    - Remove CustomerID
    """
    df = df.copy()

    # 1. Converter TotalCharges (que vem como string/object) para float
    # O 'errors=coerce' transforma espaços vazios em NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 2. Tratar valores nulos
    null_count = df['TotalCharges'].isnull().sum()
    if null_count > 0:
        logger.info(f"Preenchendo {null_count} valores nulos em TotalCharges com 0")
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Remover customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 4. Codificar variável alvo (Churn) para numérico
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    logger.info("Limpeza de dados concluída.")
    return df


def split_data(df: pd.DataFrame, target_column: str = 'Churn', test_size: float = 0.2):
    """ Divide os dados em treino e teste."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Dados divididos: Treino={X_train.shape}, Teste={X_test.shape}")
    return X_train, X_test, y_train, y_test