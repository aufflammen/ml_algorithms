import numpy as np
import pandas as pd

from .metrics import get_score, mean_squared_error, log_loss


class BaseLinear:

    def __init__(
        self, 
        n_iter=100, 
        learning_rate=.1, 
        metric=None,
        reg = None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=None
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.weights = None
        self.best_score = None
        self._need_convert_logits = False
        self._score = None if metric is None else get_score(self.metric)

    def _verbose_view(self, y_true, y_pred, loss_penalty, step, lr) -> str:
        loss = self._loss(y_true, y_pred) + loss_penalty
        if self.metric:
            # Если выполняется задача классификации и метрика подразумевает работу
            # с метками класса, а не логитами, то производим конвертацию
            if self._need_convert_logits:
                y_pred = self._logits_to_class(y_pred)
            score = self._score(y_true, y_pred)
            score_str = f' | {self.metric}: {score:.4f}'
        else:
            score_str = ''
        print(f'{step} | lr: {lr:.4f} | loss: {loss:.4f}' + score_str)

    @staticmethod
    def _to_numpy_with_bias(X: pd.DataFrame, y: pd.Series=None):
        X = X.to_numpy()
        # Добавляем столбец с единицами для включения bias
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        if y is not None:
            y = y.to_numpy()
            return X, y
        else:
            return X

    def _get_sample_size(self, n_samples: int) -> int:
        if self.sgd_sample and self.sgd_sample <= 1:
            return int(n_samples * self.sgd_sample)
        else:
            return self.sgd_sample or n_samples

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False) -> None:
        X, y = self._to_numpy_with_bias(X, y)
        n_samples, n_features = X.shape
        # Инициализируем вектор весов
        self.weights = np.ones(n_features)
        # Вычисляем количество элементов для sgd
        sample_size = self._get_sample_size(n_samples)
        np.random.seed(self.random_state)
        # Цикл обучения
        for i in range(1, self.n_iter+1):
            sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            X_sample, y_sample = X[sample_idx], y[sample_idx]
            # Prediction
            y_pred = self._get_predict(X_sample)
            # Regularization
            loss_penalty, grad_penalty = getattr(self, '_reg_' + self.reg)() if self.reg else (0, 0)
            # Gradient descent
            dw = self._get_gradient(X_sample, y_sample, y_pred, sample_size, grad_penalty)
            # Проверяем - lr является функцией или числом
            lr = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
            self.weights -= lr * dw

            # Отображение лога обучения
            if verbose and (i % verbose == 0 or i == 1 or i == self.n_iter):
                self._verbose_view(y_sample, y_pred, loss_penalty, i, lr)

        # Рассчитываем метрику после завершения обучения
        if self.metric:
            y_pred = self._get_predict(X)
            if self._need_convert_logits:
                y_pred = self._logits_to_class(y_pred)
            self.best_score = self._score(y, y_pred)

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

    def get_coef(self) -> np.ndarray:
        if self.weights is not None:
            return self.weights[1:]

    def get_best_score(self) -> float:
        return self.best_score

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}: '
            f'n_iter={self.n_iter}, learning_rate={self.learning_rate}'
        )


class LinearRegression(BaseLinear):
    """
    Parameters
    ----------
    n_iter : float, default=100
    
    learning_rate : function or float, default=0.1
        The function can be something like this: lambda iter: 0.5 * (0.85 ** iter).
    
    metric : {'mae', 'mse', 'rmse', 'mape', 'r2', None}, default=None

    reg : {'l1', 'l2', 'elasticnet', None}, default=None

    l1 : float, default=0

    l2 : float, default=0

    sgd_sample : None, int or float, default=None
        - If int, values must be in the range `(1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]`.

    random_state : None, int, default=None
    """

    def __init__(
        self, 
        n_iter=100, 
        learning_rate=.1, 
        metric=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=None
    ):
        super().__init__(
            n_iter=n_iter, 
            learning_rate=learning_rate, 
            metric=metric,
            reg=reg,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            sgd_sample=sgd_sample,
            random_state=random_state
        )
        self._loss = mean_squared_error

    @staticmethod
    def _get_gradient(X, y_true, y_pred, sample_size, grad_penalty) -> float:
        return 2 / sample_size * X.T @ (y_pred - y_true) + grad_penalty

    def _get_predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = self._to_numpy_with_bias(X)
        return self._get_predict(X)


class LogisticRegression(BaseLinear):
    """
    Parameters
    ----------
    n_iter : float, default=100
    
    learning_rate : function or float, default=0.1
        The function can be something like this: lambda iter: 0.5 * (0.85 ** iter).
    
    metric : {'logloss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', None}, default=None

    reg : {'l1', 'l2', 'elasticnet', None}, default=None

    l1 : float, default=0

    l2 : float, default=0

    sgd_sample : None, int or float, default=None
        - If int, values must be in the range `(1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]`.

    random_state : None, int, default=None
    """
    
    def __init__(
        self, 
        n_iter=100, 
        learning_rate=.1, 
        metric=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=None
    ):
        super().__init__(
            n_iter=n_iter, 
            learning_rate=learning_rate, 
            metric=metric,
            reg=reg,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            sgd_sample=sgd_sample,
            random_state=random_state
        )
        self._need_convert_logits = False if metric in {'roc_auc', 'logloss'} else True
        self._loss = log_loss

    @staticmethod
    def _sigmoid(x) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _get_gradient(X, y_true, y_pred, sample_size, grad_penalty) -> float:
        return 1 / sample_size * (y_pred - y_true) @ X + grad_penalty

    @staticmethod
    def _logits_to_class(y_pred: np.ndarray) -> np.ndarray:
        return (y_pred > .5).astype(int)

    def _get_predict(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._to_numpy_with_bias(X)
        return self._get_predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred_proba = self.predict_proba(X)
        return self._logits_to_class(y_pred_proba)