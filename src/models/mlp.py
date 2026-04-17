import torch
import torch.nn as nn

class ChurnMLP(nn.Module):
    """
    Arquitetura da Rede Neural MLP para predição de Churn.
    """
    def __init__(self, input_dim: int):
        super(ChurnMLP, self).__init__()

        self.model = nn.Sequential(
            # Camadas 1: Entrada -> 64 neurônios
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Camada 2: 64 -> 32 neurônios
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Camada 3: 32 -> 16 neurônios
            nn.Linear(32, 16),
            nn.ReLU(),

            # Saída: 16 -> 1 neurônio (saída linear para Logits)
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)