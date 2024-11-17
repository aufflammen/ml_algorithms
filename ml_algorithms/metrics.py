import numpy as np


class ClassificationMetric:

    @staticmethod
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

    @staticmethod
    def logloss(y_true, y_pred_logits):
        eps = 1e-12
        y_pred_logits = np.clip(y_pred_logits, eps, 1)
        return -np.mean(y_true * np.log(y_pred_logits) + (1 - y_true) * np.log(1 - y_pred_logits))

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        tn, fp, fn, tp = ClassificationMetric.confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    @staticmethod
    def recall(y_true, y_pred):
        tn, fp, fn, tp = ClassificationMetric.confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    @staticmethod
    def f1(y_true, y_pred):
        tn, fp, fn, tp = ClassificationMetric.confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    def roc_auc(y_true, y_pred_proba):
        sorted_indices = np.argsort(y_pred_proba)[::-1]
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


class RegressionMetric:

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    @staticmethod
    def rmse(y_true, y_pred):
        return self.mse(y_true, y_pred)**.5

    @staticmethod
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def r2(y_true, y_pred):
        return 1 - (np.mean((y_true - y_pred)**2) / np.var(y_true))


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