import torch
import numpy as np

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