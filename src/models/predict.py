import torch
import pandas as pd
from src.models.mlp import ChurnMLP
from src.features.build_features import transform_features
from src.config import MODEL_DIR, MODEL_PATH

def load_model(input_dim):
    model = ChurnMLP(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def predict(df: pd.DataFrame):

    X = transform_features(df)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    model = load_model(X_tensor.shape[1])

    with torch.no_grad():
        probs = torch.sigmoid(model(X_tensor)).numpy()

    return probs