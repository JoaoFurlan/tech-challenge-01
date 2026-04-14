import os
from pathlib import Path

# Caminho base do projeto (onde está o README)
BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas de Dados
DATA_DIR = BASE_DIR/"data"
RAW_DATA_PATH = DATA_DIR/"raw"/"telco_customer_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR/"processed"/"telco_clean.csv"

# Pastas de Modelos e Artefatos
MODEL_DIR = BASE_DIR/"models"
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR}/mlflow.db"

# Hiperparâmetros Globais
RANDOM_STATE = 42

