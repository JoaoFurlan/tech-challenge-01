from sklearn.metrics import f1_score, roc_auc_score, recall_score

def evaluate(y_true, y_prob, threshold=0.3):

    y_pred = (y_prob >= threshold).astype(int)

    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = None

    return {
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc
    }