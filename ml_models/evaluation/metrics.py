from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

def compute_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            pass
    
    return metrics
