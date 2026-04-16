import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow

from src.models.mlp import ChurnMLP
from src.utils.train_utils import EarlyStopping

def train_model(X_train, y_train, X_val, y_val, model_path):

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    # 1. Definir o peso baseado no desbalanceamento (aprox 3 para 1)
    pos_weight = torch.tensor([3.0])

    model = ChurnMLP(input_dim=X_train.shape[1])

    # 2. Passar o peso para a função de perda
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    early_stopping = EarlyStopping(patience=7, path=model_path)

    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load(model_path))

    return model