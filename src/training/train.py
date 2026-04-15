import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from src.utils.logger import get_logger
from src.config import RAW_DATA_PATH, MLFLOW_TRACKING_URI, MODEL_DIR, RANDOM_STATE
from src.data.preprocess import prepare_data_pipeline
from src.models.mlp import ChurnMLP
from src.utils.train_utils import EarlyStopping

logger = get_logger(__name__)

def train_model():
    # Garante que a pasta de modelos existe antes de começar
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Churn_Prediction_PyTorch")

    with mlflow.start_run():
        # 1. Carregar dados
        df_raw = pd.read_csv(RAW_DATA_PATH)
        X_train_full, X_test, y_train_full, y_test = prepare_data_pipeline(df_raw)

        # 2. Split de Validação (Replicando o notebook: 20% do que sobrou do treino)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
        )

        # Converter para Tensores (Treino, Validação e Teste)
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # DataLoaders
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
        
        # Configurações
        model = ChurnMLP(input_dim=X_train.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Inicializar Early Stopping
        checkpoint_path = MODEL_DIR / "mlp_churn_best.pt"
        early_stopping = EarlyStopping(patience=7, path=checkpoint_path)

        # 3. Loop de Treinamento
        epochs = 100 # Podemos aumentar as épocas pois o Early Stopping vai parar antes
        for epoch in range(epochs):
            # FASE DE TREINO
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)

            # FASE DE VALIDAÇÃO
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            # Logs no MLflow
            mlflow.log_metrics({"train_loss": avg_train_loss, "val_loss": val_loss}, step=epoch)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f}")

            # 4. Checar Early Stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping ativado na época {epoch+1}")
                break

        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_probs = torch.sigmoid(test_logits).numpy()
            # Threshold de 0.3 conforme definido na estratégia do notebook
            test_preds = (test_probs >= 0.3).astype(int) 

        # Cálculo das métricas para o Print e MLflow
        f1 = f1_score(y_test, test_preds)
        roc_auc = roc_auc_score(y_test, test_probs)
        
        report_text = classification_report(y_test, test_preds)
        report_dict = classification_report(y_test, test_preds, output_dict=True)
        accuracy = report_dict['accuracy']  # <--- Aqui pegamos a acurácia
        recall_churn = report_dict['1']['recall']
            
        print("\nRelatório de Classificação:\n", report_text)

        # Logs de métricas finais no MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("recall_churn", recall_churn)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log do modelo no MLflow
        mlflow.pytorch.log_model(model, "model")
        logger.info("Treinamento e registro no MLflow concluídos com sucesso.")


if __name__ == "__main__":
    train_model()