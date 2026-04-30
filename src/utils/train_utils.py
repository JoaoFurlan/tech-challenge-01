import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


class EarlyStopping:
    """
    Interrompe o treinamento se a perda de validação
    não melhorar após um intervalo (patience).
    """
    def __init__(self, patience=5, min_delta=0, path='model_checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Salva o modelo quando a perda de validação diminui."""
        torch.save(model.state_dict(), self.path)




def log_confusion_matrix(y_true, probs, threshold=0.3):
    """
    Gera uma matriz de confusão com a legenda em uma coluna separada à direita,
    garantindo que não haja sobreposição.
    """
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = [[f"{labels[i][j]}\n{cm[i][j]}" for j in range(2)] for i in range(2)]

    # 1. Criar a figura e definir o Grid
    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.4)

    # 2. Eixo da Esquerda (Matriz)
    ax0 = plt.subplot(gs[0])
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax0,
        cbar=False
    )
    ax0.set_title(f"Matriz de Confusão (Threshold = {threshold})", fontsize=12, pad=15)
    ax0.set_xlabel("Previsto (Predicted)", fontsize=10)
    ax0.set_ylabel("Real (Actual)", fontsize=10)

    # 3. Eixo da Direita (Legenda - Invisível)
    ax1 = plt.subplot(gs[1])
    ax1.axis('off') # Esconde os eixos da área da legenda

    legend_text = (
        """Legenda Detalhada:
        
            TN (True Negatives):
            O modelo previu corretamente
            que o cliente FICARIA.

            FN (False Negatives):
            O modelo errou ao dizer que o
            cliente ficaria, mas ele SAIU.

            TP (True Positives):
            O modelo previu corretamente
            que o cliente SAIRIA (Churn).

            FP (False Positives):
            O modelo previu que o cliente
            sairia, mas ele FICOU."""
    )

    # Adicionamos o texto dentro do eixo ax1 (invisível)
    # x=0 e y=0.5 significa que o texto começa no início do eixo da direita
    ax1.text(0.0, 0.5, legend_text,
             va="center", ha="left",
             fontsize=9, linespacing=1.4,
             bbox={"facecolor":"white", "alpha":0.8, "edgecolor":"lightgray", "pad":10})

    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, "confusion_matrix_final.png")

    #plt.tight_layout()

    # Salvar e enviar ao MLflow
    plt.savefig(plot_path, bbox_inches='tight')

    mlflow.log_artifact(plot_path)

    plt.close(fig)
