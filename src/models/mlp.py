import torch
import torch.nn as nn

class ChurnMLP(nn.Module):
    """
    Arquitetura da Rede Neural MLP para predição de Churn.
    """
    def __init__(self, input_dim: int):
        super(ChurnMLP, self).__init__()

        # Camadas densas (Linear) seguidas de ativação ReLU e Dropout para evitar overfitting
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x