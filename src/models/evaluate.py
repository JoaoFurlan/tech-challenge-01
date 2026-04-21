from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score 

def evaluate(y_true, y_prob, threshold=0.3):
    """
    Calcula as principais métricas de classificação para o modelo de Churn.
    
    A função utiliza probabilidades para calcular o ROC-AUC e valores binários 
    (baseados no threshold) para as demais métricas.

    Args:
        y_true (array-like): Valores reais do target (0 ou 1).
        y_prob (array-like): Probabilidades previstas pelo modelo (saída da Sigmoid).
        threshold (float): Limite de decisão para converter probabilidade em classe. 
                           Padrão é 0.3 para priorizar Recall.

    Returns:
        dict: Dicionário contendo Accuracy, Precision, Recall, F1 e ROC-AUC.
    """

    # Converte as probabilidades em classes binárias (0 ou 1) com base no threshold
    y_pred = (y_prob >= threshold).astype(int)

    # O cálculo do ROC-AUC pode falhar se o conjunto de teste contiver apenas uma classe.
    # Usamos try/except para garantir que o pipeline não pare caso isso ocorra.
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        # Se falhar, definimos como None para não interromper o treinamento
        roc = None
        
    # Retorno das métricas formatadas para log e MLflow
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc
    }