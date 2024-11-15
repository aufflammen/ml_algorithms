import numpy as np


class ClassificationMetrics:

    def confusion_matrix(self, y_true, y_pred):
        labels = np.unique([y_true, y_pred])
        n_labels = len(labels)
        label_to_index = {label: index for index, label in enumerate(labels)}

        matrix = np.zeros((n_labels, n_labels), dtype=np.int64)

        for label_true, label_pred in zip(y_true, y_pred):
            i = label_to_index[label_true]
            j = label_to_index[label_pred]
            matrix[i, j] += 1

        return matrix
        
    def logloss(self, y_true, y_pred_logits):
        eps = 1e-12
        y_pred_logits = np.clip(y_pred_logits, eps, 1)
        return -np.mean(y_true * np.log(y_pred_logits) + (1 - y_true) * np.log(1 - y_pred_logits))

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def precision(self, y_true, y_pred):
        tn, fp, fn, tp = self.confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall(self, y_true, y_pred):
        tn, fp, fn, tp = self.confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def f1(self, y_true, y_pred):
        tn, fp, fn, tp = self.confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    def roc_auc(self, y_true, y_pred_proba):
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