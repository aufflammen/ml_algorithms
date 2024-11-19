import numpy as np
import pandas as pd
import random
from collections import Counter

from .decision_tree import DecisionTreeRegressor, DecisionTreeClassifier
from .metrics import get_score

class BaseRandomForest:

    def __init__(
        self, 
        n_estimators=10,
        max_features=.5,
        max_samples=.5,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        oob_score = None,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.oob_score = None if oob_score is None else get_score(oob_score)
        self.random_state = random_state
        
        self.leafs_cnt = 0
        self.fi = None # Feature importance
        self.ensemble = []
        self.oob_score_ = None

    def __str__(self):
        valid_params = set([
            'n_estimators', 
            'max_features', 
            'max_samples',  
            'max_depth', 
            'min_samples_split', 
            'max_leafs', 
            'bins',
            'random_state'
            ])

        params_print = ', '.join(
            f'{k}={v}' for k, v in self.__dict__.items() if k in valid_params
        )
        return f'{self.__class__.__name__} class: {params_print}'


class RandomForestRegressor(BaseRandomForest):

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        n_samples, n_features = X.shape
        features = X.columns.to_list()
        self.fi = Counter({k: 0 for k in X.columns})
        # Рассчитываем количество объектов и фичей, 
        # на которых будет обучаться каждое отдельное дерево
        cnt_col = round(n_features * self.max_features)
        cnt_row = round(n_samples * self.max_samples)
        
        oob_pred_sum = np.zeros(n_samples)
        oob_pred_count = np.zeros(n_samples)
        random.seed(self.random_state)

        for _ in range(self.n_estimators):
            col_indices = random.sample(features, cnt_col)
            row_indices = random.sample(range(n_samples), cnt_row)
            model = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split, 
                max_leafs=self.max_leafs, 
                bins=self.bins
            )
            # Переопределяем параметр n_samples в каждом дереве
            # для корректного подсчета важности фичей при беггинге
            model.n_samples = n_samples
            model.fit(X.loc[row_indices, col_indices], y.loc[row_indices])
            self.leafs_cnt += model.leafs_cnt
            self.fi += Counter(model.fi)
            self.ensemble.append(model)
            # Out of bag
            if self.oob_score is not None:
                oob_indices = list(set(range(n_samples)) - set(row_indices))
                pred = model.predict(X.loc[oob_indices])

                oob_pred_count[oob_indices] += 1
                oob_pred_sum[oob_indices] += pred

        if self.oob_score is not None:
            oob_inidices = np.where(oob_pred_count > 0)[0]
            oob_pred_mean = oob_pred_sum[oob_inidices] / oob_pred_count[oob_inidices]
            self.oob_score_ = self.oob_score(y.to_numpy()[oob_inidices], oob_pred_mean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.zeros(X.shape[0])
        for model in self.ensemble:
            pred += model.predict(X)
        pred /= self.n_estimators
        
        return pred
