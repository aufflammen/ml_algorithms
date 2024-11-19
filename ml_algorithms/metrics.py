import numpy as np


# ------------------------------------
# Classification metrics
# ------------------------------------

def _precision_recall_fscore(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def confusion_matrix(y_true, y_pred):
    labels = np.unique([y_true, y_pred])
    n_labels = len(labels)
    label_to_index = {label: index for index, label in enumerate(labels)}

    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)

    for label_true, label_pred in zip(y_true, y_pred):
        i = label_to_index[label_true]
        j = label_to_index[label_pred]
        matrix[i, j] += 1

    return matrix

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    return _precision_recall_fscore(y_true, y_pred)[0]

def recall_score(y_true, y_pred):
    return _precision_recall_fscore(y_true, y_pred)[1]

def f1_score(y_true, y_pred):
    return _precision_recall_fscore(y_true, y_pred)[2]

def log_loss(y_true, y_pred):
    eps = 1e-12
    y_pred_proba = np.clip(y_pred, eps, 1)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def roc_auc_score(y_true, y_pred):
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    
    tp = auc = 0

    for y in y_true_sorted:
        if y == 1:
            tp += 1
        else:
            auc += tp  # Добавляем количество TP для каждого FP

    # Нормируем площадь
    return auc / (n_pos * n_neg)


# ------------------------------------
# Regression metrics
# ------------------------------------

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**.5

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2_score(y_true, y_pred):
    return 1 - (np.mean((y_true - y_pred)**2) / np.var(y_true))


# ------------------------------------
# Distance
# ------------------------------------

class Distance:

    @staticmethod
    def euclidean(arr1, arr2):
        return np.sqrt(np.sum(np.square(arr1 - arr2), axis=-1))

    @staticmethod
    def manhattan(arr1, arr2):
        return np.sum(np.abs(arr1 - arr2), axis=-1)

    @staticmethod
    def chebyshev(arr1, arr2):
        return np.max(np.abs(arr1 - arr2), axis=-1)

    @staticmethod
    def cosine(arr1, arr2):
        l2_norm = lambda x: np.sqrt(np.sum(np.square(x), axis=-1))
        return 1 - np.sum(arr1 * arr2, axis=-1) / (l2_norm(arr1) * l2_norm(arr2))


_SCORERS = dict(
    # classification
    confusion_matrix=confusion_matrix,
    logloss=log_loss,
    accuracy=accuracy_score,
    precision=precision_score,
    recall=recall_score,
    f1=f1_score,
    roc_auc=roc_auc_score,
    # regression
    mae=mean_absolute_error,
    mse=mean_squared_error,
    rmse=root_mean_squared_error,
    mape=mean_absolute_percentage_error,
    r2=r2_score
)


def get_score_name():
    return _SCORERS.keys()

def get_score(metric):
    if isinstance(metric, str):
        try:
            score = _SCORERS[metric]
        except:
            raise ValueError(
                f"'{metric}' is not a valid scoring value.\n"
                f"Parameter must be a str among {get_score_name()}."
            )
    elif callable(metric):
        score = metric
    else:
        raise ValueError(f'Parameter must be either a str or func.')
    return score