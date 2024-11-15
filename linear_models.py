import numpy as np
import pandas as pd
import random

from .metrics import *



class BaseLinearEstimator:

    def _reg_l1(self):
        loss_penalty = self.l1_coef * np.sum(np.abs(self.weights))
        grad_penalty = self.l1_coef * np.sign(self.weights)
        return loss_penalty, grad_penalty

    def _reg_l2(self):
        loss_penalty = self.l2_coef * np.sum((self.weights)**2)
        grad_penalty = self.l2_coef * 2 * self.weights
        return loss_penalty, grad_penalty

    def _reg_elasticnet(self):
        loss_penalty_l1, grad_penalty_l1 = self._reg_l1()
        loss_penalty_l2, grad_penalty_l2 = self._reg_l2()
        return loss_penalty_l1 + loss_penalty_l2, grad_penalty_l1 + grad_penalty_l2

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.best_metric
    

class MyLogReg(ClassificationMetrics, BaseLinearEstimator):
    
    def __init__(
        self, 
        n_iter=100, 
        learning_rate=.1, 
        metric=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=42
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.weights = None
        self.best_metric = None
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def _get_metric(self, y_true, y_pred_logits):
        if self.metric == 'roc_auc':
            return self.roc_auc(y_true, y_pred_logits)
        else:
            y_pred = (y_pred_logits > .5).astype(int)
            return getattr(self, self.metric)(y_true, y_pred)

    def _verbose_view(self, y_true, y_pred_logits, loss_penalty, step, lr):
        loss = self.logloss(y_true, y_pred_logits) + loss_penalty
        if self.metric:
            metric = self._get_metric(y_true, y_pred_logits)
            print(f'{step} | lr: {lr:.4f} | loss: {loss:.4f} | {self.metric}: {metric:.4f}')
        else:
            print(f'{step} | lr: {lr:.4f} | loss: {loss:.4f}')
        
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False) -> None:
        X, y = X.to_numpy(), y.to_numpy()
        n_samples, n_features = X.shape
        # Добавляем столбец с единицами для включения bias
        X = np.hstack([np.ones((n_samples, 1)), X])
        # Инициализируем вектор весов
        self.weights = np.ones(n_features + 1)
        # Вычисляем количество элементов для sgd
        sample_size = int(n_samples * self.sgd_sample) if self.sgd_sample and self.sgd_sample <= 1 else (self.sgd_sample or n_samples)
        random.seed(self.random_state)

        for i in range(1, self.n_iter+1):
            # sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            sample_idx = random.sample(range(n_samples), sample_size)
            X_sample, y_sample = X[sample_idx], y[sample_idx]
            # Prediction
            y_pred_logits = self._sigmoid(X_sample @ self.weights)
            # Regularization
            loss_penalty, grad_penalty = getattr(self, '_reg_' + self.reg)() if self.reg else (0, 0)
            # Gradient descent
            dw = 1 / n_samples * (y_pred_logits - y_sample) @ X_sample + grad_penalty
            # Проверяем lr функция или число
            lr = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
            self.weights -= lr * dw

            # Отображение лога обучения
            if verbose and (i % verbose == 0 or i == 1 or i == self.n_iter):
                self._verbose_view(y_sample, y_pred_logits, loss_penalty, i, lr)

        # Рассчитываем наилучшую метрику после завершения обучения
        if self.metric:
            y_pred_logits = self._sigmoid(X @ self.weights)
            self.best_metric = self._get_metric(y, y_pred_logits)

    def predict_proba(self, X: pd.DataFrame):
        X = X.to_numpy()
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(X @ self.weights)

    def predict(self, X: pd.DataFrame):
        logits = self.predict_proba(X)
        return (logits > .5).astype(int)

    def __str__(self):
        return f'{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'