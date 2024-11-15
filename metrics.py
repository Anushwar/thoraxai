# metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_pred):
    num_classes = y_true.shape[1]
    aucs = []

    for i in range(num_classes):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    return {"mean_auc": np.mean(aucs)}
