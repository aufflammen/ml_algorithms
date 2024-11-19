import numpy as np
import pandas as pd
import random
from collections import Counter

from .base import *
from .decision_tree import DecisionTreeRegressor, DecisionTreeClassifier
from .metrics import get_score


# ------------------------------------
# RandomForest
# ------------------------------------

class BaseRandomForest(BaseModel):

    def __init__(
        self, 
        n_estimators=10,
        max_features=.5,
        max_samples=.5,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        oob_score=None,
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
        self.model_task = 'regression'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        n_samples, n_features = X.shape
        features = X.columns.to_list()
        self.fi = Counter({k: 0 for k in X.columns})
        # Рассчитываем количество объектов и фичей, 
        # на которых будет обучаться каждое отдельное дерево
        cnt_col = round(n_features * self.max_features)
        cnt_row = round(n_samples * self.max_samples)
        random.seed(self.random_state)

        if self.oob_score:
            oob_pred_sum = np.zeros(n_samples)
            oob_pred_count = np.zeros(n_samples)
            if self.model_task == 'regression':
                pred_method = 'predict'
            elif self.model_task == 'classification':
                pred_method = 'predict_proba'

        for _ in range(self.n_estimators):
            col_indices = random.sample(features, cnt_col)
            row_indices = random.sample(range(n_samples), cnt_row)
            model = self._tree_model()
            # Переопределяем параметр n_samples в каждом дереве
            # для корректного подсчета важности фичей при беггинге
            model.n_samples = n_samples
            model.fit(X.loc[row_indices, col_indices], y.loc[row_indices])
            self.leafs_cnt += model.leafs_cnt
            # Подсчет важности фичей
            self.fi += Counter(model.fi)
            self.ensemble.append(model)
            # Out of bag
            if self.oob_score is not None:
                oob_indices = list(set(range(n_samples)) - set(row_indices))
                pred = getattr(model, pred_method)(X.loc[oob_indices])
                oob_pred_count[oob_indices] += 1
                oob_pred_sum[oob_indices] += pred

        if self.oob_score is not None:
            self._compute_oob_score(y, oob_pred_sum, oob_pred_count)
            
    def _compute_oob_score(
        self, 
        y: pd.Series, 
        oob_pred_sum: np.ndarray, 
        oob_pred_count: np.ndarray
    ) -> None:
            oob_inidices = np.where(oob_pred_count > 0)[0]
            oob_pred_mean = oob_pred_sum[oob_inidices] / oob_pred_count[oob_inidices]
            if all((
                self.model_task == 'classification', 
                self.oob_score.__name__ not in {'roc_auc_score', 'log_loss'}
            )):
                oob_pred_mean = self._logits_to_class(oob_pred_mean)
            # self.oob_score_ = self.oob_score(y.to_numpy()[oob_inidices], oob_pred_mean)
            self.oob_score_ = y.to_numpy()[oob_inidices], oob_pred_mean
        

    def _get_predict(self, X: pd.DataFrame, method: str = 'predict') -> np.ndarray:
        """
        type : {'predict', 'predict_proba'}, default='predict'
        """
        pred = np.zeros(X.shape[0])
        for model in self.ensemble:
            pred += getattr(model, method)(X)
        return pred / self.n_estimators


class RandomForestRegressor(BaseRandomForest):

    def _tree_model(self):
        model = DecisionTreeRegressor(
            max_depth=self.max_depth, 
            min_samples_split=self.min_samples_split, 
            max_leafs=self.max_leafs, 
            bins=self.bins
        )
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._get_predict(X, method='predict')


class RandomForestClassifier(BaseRandomForest):
    
    def __init__(
        self, 
        n_estimators=10,
        max_features=.5,
        max_samples=.5,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16,
        oob_score=None,
        random_state=42,
        criterion='entropy',
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_leafs=max_leafs,
            bins=bins,
            oob_score=oob_score,
            random_state=random_state
        )
        self.criterion=criterion
        self.model_task = 'classification'

    def _tree_model(self):
        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split, 
            max_leafs=self.max_leafs, 
            bins=self.bins
        )
        return model

    @staticmethod
    def _logits_to_class(y_pred: np.ndarray) -> np.ndarray:
        return (y_pred > .5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._get_predict(X, method='predict_proba')

    def predict(self, X: pd.DataFrame, type='mean') -> np.ndarray:
        """
        type : {'mean', 'vote'}, default='mean'
        """
        if type == 'mean':
            pred = self._get_predict(X, method='predict_proba')
        elif type == 'vote':
            pred = self._get_predict(X, method='predict')

        return self._logits_to_class(pred)


# ------------------------------------
# Boosting
# ------------------------------------

class BaseGradientBoosting(BaseModel):
    pass


#GradientBoostingRegressor
class MyBoostReg(BaseGradientBoosting):

    def __init__(
        self, 
        n_estimators=10,
        learning_rate=.1,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=16
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins


