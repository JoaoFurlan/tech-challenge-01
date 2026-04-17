import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
from sklearn.metrics import confusion_matrix

class EarlyStopping:
    """Interrompe o treinamento se a perda de validação não melhorar após um intervalo (patience)."""
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
    Gera uma matriz de confusão estilizada com legendas detalhadas 
    e faz o upload para o MLflow.
    """
    # Converter probabilidades em predições binárias
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # Criar as anotações customizadas (TN, FP, FN, TP)
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(f"{labels[i][j]}\n{cm[i][j]}")
        annot.append(row)

    # Configurar a figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm, 
        annot=annot, 
        fmt="", 
        cmap="Blues", 
        xticklabels=["No Churn", "Churn"], 
        yticklabels=["No Churn", "Churn"], 
        ax=ax
    )

    ax.set_title(f"Matriz de Confusão (Threshold = {threshold})", fontsize=14, pad=20)
    ax.set_xlabel("Previsto (Predicted)", fontsize=12)
    ax.set_ylabel("Real (Actual)", fontsize=12)

    # Adicionar a legenda explicativa no rodapé da imagem
    plt.figtext(
        0.5, -0.1, 
        "Legenda Detalhada:\n"
        "• TN (True Negatives): O modelo previu corretamente que o cliente FICARIA.\n"
        "• FN (False Negatives): O modelo errou ao dizer que o cliente ficaria, mas ele SAIU.\n"
        "• TP (True Positives): O modelo previu corretamente que o cliente SAIRIA (Churn).\n"
        "• FP (False Positives): O modelo previu que o cliente sairia, mas ele FICOU.",
        ha="center", 
        fontsize=10, 
        bbox={"facecolor":"white", "alpha":0.8, "edgecolor":"gray", "pad":10}
    )

    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, "confusion_matrix_final.png")

    plt.tight_layout()

    # Salvar e enviar ao MLflow
    plt.savefig(plot_path, bbox_inches='tight')

    mlflow.log_artifact(plot_path)
    
    plt.close(fig)