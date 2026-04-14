import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

from src.utils.logger import get_logger
from src.config import RAW_DATA_PATH, MLFLOW_TRACKING_URI
from src.data.preprocess import prepare_data_pipeline
from src.models.mlp import ChurnMLP

logger = get_logger(__name__)

def train_model():
    # 1. Configurar o MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Churn_Prediction_PyTorch")

    with mlflow.start_run() as run:
        logger.info("Iniciando run no MLflow e carregando dados...")

        # 2. Carregar e preparar os dados
        df_raw = pd.read_csv(RAW_DATA_PATH)
        X_train, X_test, y_train, y_test = prepare_data_pipeline(df_raw)

        # 3. Converter para tensores do pyTorch
        logger.info("Convertendo dados para tensores do PyTorch...")
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # 4. Configurar DataLoaders e Hiperparâmetros
        batch_size = 64
        epochs = 30
        learning_rate = 0.001

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

        # Log dos hiperparâmetros no MLflow
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate})

        # 5. Instaciar o modelo, Loss e Otimizador
        input_dim = X_train.shape[1]
        model = ChurnMLP(input_dim)

        # BCEWithlogitsLoss é ideal para classificação binária (combina Sigmoid e BCELoss)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 6. Loop de Treinamento
        logger.info("Iniciando loop de treinamento...")
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            # Log da perda a cada época
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # 7. Avaliação Simples (Acurácia no Teste)
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            # Como usamos BCEWithLogits, aplicamos a sigmoid para pegar a probabilidade
            predictions = torch.sigmoid(test_outputs).round()
            correct = (predictions == y_test_t).sum().item()
            accuracy = correct / y_test_t.size(0)

        logger.info(f"Acurácia no conjunto de teste: {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)

        # 8. Salvar o modelo PyTorch no MLflow
        mlflow.pytorch.log_model(model, "pytorch_model")
        logger.info("Treinamento finalizado. Modelo salvo no MLflow.")


if __name__ == "__main__":
    train_model()