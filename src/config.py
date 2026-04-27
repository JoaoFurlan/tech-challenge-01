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
MODEL_PATH = MODEL_DIR / "mlp_churn_best.pt"

# MLflow com variável de ambiente
# 1. Tenta ler do sistema (Docker/Server)
# 2. Se não existir, monta o caminho padrão dinâmico
DEFAULT_MLFLOW_URI = f"sqlite:///{BASE_DIR.as_posix()}/mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)


TARGET = "Churn"

# Hiperparâmetros Globais
RANDOM_STATE = 42
CHURN_THRESHOLD = 0.3

# Cores para o terminal
C_GREEN = "\033[32m"
C_CYAN = "\033[36m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
