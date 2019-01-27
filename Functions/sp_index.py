import numpy as np
from sklearn.metrics import recall_score

def sp_index(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    recall = recall_score(y_true, y_pred, average=None)
    sp = np.sqrt(np.sum(recall) / num_classes *
                 np.power(np.prod(recall), 1.0 / float(num_classes)))
    return sp


