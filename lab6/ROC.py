import numpy as np

def auc_roc(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)
    sorted_labels = y_true[sorted_indices]
    num_positives = np.sum(sorted_labels == 1)
    num_negatives = np.sum(sorted_labels == 0)
    
    tpr = []
    fpr = []
    
    tp, fp = 0, 0
    
    for i in range(len(sorted_indices) - 1, -1, -1):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / num_positives)
        fpr.append(fp / num_negatives)
    
    return fpr, tpr

def auc_pr(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    precision = []
    recall = []
    
    tp, fp = 0, 0
    
    for i in range(len(sorted_indices)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        precision.append(tp / (tp + fp))
        recall.append(tp / np.sum(sorted_labels == 1))
    
    return recall, precision
