import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.middleware.logger import get_logger
from src.models.mlp import ChurnMLP
from src.utils.train_utils import EarlyStopping

logger = get_logger(__name__)

def train_model(X_train, y_train, X_val, y_val, model_path):

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    # 1. Definir o peso baseado no desbalanceamento
    pos_weight = torch.tensor([1.0])

    model = ChurnMLP(input_dim=X_train.shape[1])

    # 2. Passar o peso para a função de perda
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    early_stopping = EarlyStopping(patience=10, path=model_path)


    for epoch in range(100):
        # --- TREINO ---
        model.train()
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)

        # --- VALIDAÇÃO ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                running_val_loss += loss_val.item()

        epoch_val_loss = running_val_loss / len(val_loader)

        # --- LOGS E EARLY STOPPING ---
        early_stopping(epoch_val_loss, model)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

        if early_stopping.early_stop:
            logger.info(f"Early stopping acionado na época {epoch}")
            break

    model.load_state_dict(torch.load(model_path))

    return model
