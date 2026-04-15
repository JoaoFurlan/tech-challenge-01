import pandas as pd
import mlflow
import torch
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_PATH, MODEL_PATH, MLFLOW_TRACKING_URI
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.build_features import fit_transform_features, transform_features
from src.models.train import train_model
from src.models.evaluate import evaluate
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline():
    """
    Pipeline completo:
    1. Carrega dados
    2. Limpa dados
    3. Split treino/validação
    4. Feature engineering
    5. Treina modelo
    """
    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Churn_Prediction_MLP")

    with mlflow.start_run():
        # 1. Load e 2. Clean
        df = load_data(RAW_DATA_PATH)
        df = clean_data(df)

        # 3. Separar target e 4. Split
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Features
        X_train = fit_transform_features(X_train)
        X_val = transform_features(X_val)

        # 6. Treino
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_path=MODEL_PATH
        )

        # 7. Avaliação
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
            # Obter probabilidades (usando sigmoid pois a saída do modelo é linear)
            y_prob = torch.sigmoid(model(X_val_t)).numpy().flatten()
        
        metrics = evaluate(y_val, y_prob)
        
        logger.info(f"Métricas Finais - Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

        # 8. Log no MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "MLP")
        mlflow.pytorch.log_model(model, "model")

        logger.info(f"Pipeline finalizado. Métricas: {metrics}")

    