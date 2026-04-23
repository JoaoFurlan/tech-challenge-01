import random

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import (
    C_BOLD,
    C_CYAN,
    C_GREEN,
    C_RESET,
    CHURN_THRESHOLD,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.build_features import fit_transform_features, transform_features
from src.middleware.logger import get_logger
from src.models.evaluate import evaluate
from src.models.train import train_model
from src.utils.train_utils import log_confusion_matrix

logger = get_logger(__name__)



def run_training_pipeline():
    """
    Pipeline completo:
    1. Carrega dados
    2. Limpa dados
    3. Separar Target
    4. Split treino/validação/teste
    5. Feature engineering
    6. Treina modelo
    7. Avaliação do modelo
    8. Log no MLflow
    """
    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Churn_Prediction_MLP")

    # Trava aleatoriedade
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    with mlflow.start_run():
        # 1. Load e 2. Clean
        df = load_data(RAW_DATA_PATH)
        df = clean_data(df)

        # 3. Separar target
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        # 4. Split treino, validação e teste
        # Primeiro split: separa o conjunto de teste (20%)
        # e o conjunto completo para treino/validação (80%)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
                                            X,
                                            y,
                                            test_size=0.2,
                                            random_state=RANDOM_STATE,
                                            stratify=y)

        # Segundo split: subdivide o conjunto completo em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
                                            X_train_full,
                                            y_train_full,
                                            test_size=0.2,
                                            random_state=RANDOM_STATE,
                                            stratify=y_train_full)


        # 5. Features
        X_train = fit_transform_features(X_train)
        X_val = transform_features(X_val)
        X_test = transform_features(X_test)

        # 6. Treino (utiliza o conjunto de validação para o early stopping)
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_path=MODEL_PATH
        )


        # 7. Avaliação (agora utilizando o conjunto de teste isolado)
        # Carrega os pesos salvos pelo EarlyStopping antes de avaliar
        device = torch.device("cpu")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
            # Obter probabilidades (usando sigmoid pois a saída do modelo é linear)
            y_prob = torch.sigmoid(model(X_test_t)).numpy().flatten()



        metrics = evaluate(y_test, y_prob, threshold=CHURN_THRESHOLD)


        # 8. Log no MLflow
        log_confusion_matrix(y_test, y_prob, threshold=CHURN_THRESHOLD)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "MLP", CHURN_THRESHOLD)
        mlflow.pytorch.log_model(model, "model")

        # Log formatado
        logger.info(f"{C_BOLD}{C_CYAN}Pipeline finalizado com sucesso!{C_RESET}")
        logger.info(f"{C_CYAN}--- Métricas de Avaliação ---{C_RESET}")
        logger.info(f"Acurácia:  {C_GREEN}{metrics['accuracy']:.4f}{C_RESET}")
        logger.info(f"Precisão:  {C_GREEN}{metrics['precision']:.4f}{C_RESET}")
        logger.info(f"Recall:    {C_GREEN}{metrics['recall']:.4f}{C_RESET}")
        logger.info(f"F1-Score:  {C_GREEN}{metrics['f1']:.4f}{C_RESET}")
        logger.info(f"ROC-AUC:   {C_GREEN}{metrics['roc_auc']:.4f}{C_RESET}")
        logger.info(f"{C_CYAN}-----------------------------{C_RESET}")

