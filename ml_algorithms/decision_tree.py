import numpy as np
import pandas as pd

from .metrics import *


class TreeNode:
    def __init__(self):
        self.n_samples = None
        self.criterion = None
        self.information_gain = None
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = True


class BaseDecisionTree:

    # Рекурсивное построение дерева
    def _build_tree(self, node: TreeNode, X: np.ndarray, y: np.ndarray, depth: int):
        if self.tree_task == 'classification':
            values, counts = np.unique(y, return_counts=True)
            node.value = np.zeros(self.n_classes, dtype=int)
            # Для каждого уникального класса указываем количество его экземпляров
            # с учетом того, что некоторых классов может не быть в данном конкретном разбиении
            node.value[np.vectorize(self.class_to_idx.get)(values)] = counts
            # Строка выше делает то же самое, что следующий цикл:
            # for val, cnt in zip(values, counts):
            #     node.value[self.class_to_idx[val]] = cnt
        else:
            node.value = np.mean(y)

        node.n_samples = len(y)
        node.criterion = self._criterion_func(y)
        
        # Условие остановки ветвления: проверяем, достигли ли мы максимальной глубины или минимального числа листьев
        if any((
            depth >= self.max_depth, 
            self.leafs_cnt >= max(2, self.max_leafs), 
            len(y) < self.min_samples_split, 
            len(np.unique(y)) == 1 if self.tree_task == 'classification' else False
        )):
            return node

        # Выбираем наилучший признак и порог для разбиения
        feature_idx, threshold, information_gain = self._get_best_split(X, y)
        if feature_idx is None or threshold is None:
            return node

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.information_gain = information_gain
        node.is_leaf = False
        self.leafs_cnt += 1

        # Разделяем данные на левую и правую части
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        # Обновление списка важности признаков
        fi = (len(left_y) + len(right_y)) * information_gain / self.n_samples_train
        self.fi[self.features[feature_idx]] += fi 
        
        # Рекурсивно строим поддеревья для каждого разбиения
        node.left = self._build_tree(TreeNode(), left_X, left_y, depth + 1)
        node.right = self._build_tree(TreeNode(), right_X, right_y, depth + 1)

        return node

    # Поиск наилучшего разбиения
    def _get_best_split(self, X: np.ndarray, y: np.ndarray):
        best_criterion = begin_criterion = self._criterion_func(y) # энтропия до разделения
        best_feature_idx = best_threshold = best_information_gain = None
    
        for feature_idx in range(self.n_features):
            X_current_feature = X[:, feature_idx]
            # Получаем отсортированный список уникальных значений признака
            unique_feature_values = np.unique(X_current_feature)

            if self.bins:
                thresholds = self.thresholds_bins[feature_idx]
            else:
                # В качестве порогов для разбиения берем среднее между каждой парой значений
                thresholds = np.convolve(unique_feature_values, [.5, .5], mode='valid')   # аналог (a[:-1] + a[1:]) / 2
    
            for threshold in thresholds:
                left_indices = X_current_feature <= threshold
                right_indices = X_current_feature > threshold
                left_y, right_y = y[left_indices], y[right_indices]
                len_left, len_right = len(left_y), len(right_y)
                len_total = len_left + len_right
                
                if len_left == 0 or len_right == 0:
                    continue

                criterion_split = (len_left * self._criterion_func(left_y) + len_right * self._criterion_func(right_y)) / len_total

                # проверяем, уменьшает ли текущее разделение энтропию
                if best_criterion > criterion_split:
                    best_criterion = criterion_split
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_information_gain = begin_criterion - criterion_split
                    
        return best_feature_idx, best_threshold, best_information_gain

    # Вывод структуры построенного дерева
    def print_tree(self) -> str:
        # Рекурсивная функция для обхода дерева
        def traverse_print_tree(node, parent=None, depth=1):
            criterion_info = f"{self.criterion}: {node.criterion:<8.4f} {node.value.tolist() if self.tree_task == 'classification' else ''}"
            if not node.is_leaf:
                print(f"{'-' * depth + str(self.features[node.feature_idx]):<25} > {node.threshold:<10.4f} {criterion_info}")
                if node.left is not None:
                    traverse_print_tree(node.left, 'left', depth + 1)
                if node.right is not None:
                    traverse_print_tree(node.right, 'right', depth + 1)
            else:
                if self.tree_task == 'classification':
                    leaf_info = f"{'-' * depth + 'leaf_' + parent:<25} = {self.classes[np.argmax(node.value)]:<10} {criterion_info}"
                else:
                    leaf_info = f"{'-' * depth + 'leaf_' + parent:<25} = {node.value:<10.4f} {criterion_info}"
                print(leaf_info)

        return traverse_print_tree(self.root)

    def __str__(self) -> str:
        return f'{self.__class__.__name__} class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'


class DecisionTreeRegressor(BaseDecisionTree):
    """
    Parameters
    ----------
    max_depth : int, default=5

    min_samples_split : int, default=2

    max_leaf : int, default=20

    bins : int, default=None
        Values must be in the range `[3, inf)`.
    """

    def __init__(
        self, 
        max_depth=5,
        min_samples_split=2,
        max_leafs=20,
        bins=None
    ):
        self.tree_task = 'regression'
        self.criterion = 'mse'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins

        self.root = None
        self.features = None
        self.n_features = None
        self.n_samples_train = None
        self.fi = None # Feature importance
        self._criterion_func = self._mse

        if self.bins and self.bins < 3:
            raise ValueError('Parameter `bins` must be in the range `[3, inf)`')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.features = X.columns
        self.n_features = len(self.features)
        self.n_samples_train = X.shape[0]
        self.fi = {k: 0 for k in self.features}
        X, y = X.to_numpy(), y.to_numpy()

        if self.bins:
            self.thresholds_bins = []
            for feature_idx in range(self.n_features):
                self.thresholds_bins.append(np.histogram(X[:, feature_idx], bins=self.bins)[1][1:-1])
        
        self.root = self._build_tree(TreeNode(), X, y, depth=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        def traverse(node, indices):
            if node.is_leaf:
                predict[indices] = node.value
                return

            if node.left:
                left_indices = indices[X[indices, node.feature_idx] <= node.threshold]
                traverse(node.left, left_indices)
            if node.right:
                right_indices = indices[X[indices, node.feature_idx] > node.threshold]
                traverse(node.right, right_indices)
        
        # Устанавливаем такой же порядок признаков, как в обучающей выборке 
        X = X[self.features].to_numpy()
        n_samples = X.shape[0]
        predict = np.zeros(n_samples)
        traverse(self.root, np.arange(n_samples))
        return predict

    def _mse(self, sample: np.ndarray) -> float:
        return np.mean(np.square(sample - np.mean(sample)))
        

class DecisionTreeClassifier(BaseDecisionTree):
    """
    Parameters
    ----------
    max_depth : int, default=5

    criterion : {gini, entropy}, default="entropy"

    min_samples_split : int, default=2

    max_leaf : int, default=20

    bins : int, default=None
        Values must be in the range `[3, inf)`.
    """

    def __init__(
        self, 
        max_depth=5,
        criterion='entropy',
        min_samples_split=2,
        max_leafs=20,
        bins=None
    ):
        self.tree_task = 'classification'
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        
        self.root = None
        self.features = None
        self.n_features = None
        self.n_samples_train = None
        self.fi = None # Feature importance
        self.classes = None
        self.class_to_idx = None
        self.n_classes = None
        self._criterion_func = getattr(self, '_' + self.criterion)

        if self.bins and self.bins < 3:
            raise ValueError('Parameter `bins` must be in the range `[3, inf)`')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.features = X.columns
        self.n_features = len(self.features)
        self.n_samples_train = X.shape[0]
        self.fi = {k: 0 for k in self.features}
        self.classes = np.unique(y)
        self.class_to_idx = {v: k for k, v in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        X, y = X.to_numpy(), y.to_numpy()

        if self.bins:
            self.thresholds_bins = []
            for feature_idx in range(self.n_features):
                self.thresholds_bins.append(np.histogram(X[:, feature_idx], bins=self.bins)[1][1:-1])
        
        self.root = self._build_tree(TreeNode(), X, y, depth=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        def traverse(node, indices):
            if node.is_leaf:
                probs[indices] = node.value / node.n_samples
                return

            if node.left:
                left_indices = indices[X[indices, node.feature_idx] <= node.threshold]
                traverse(node.left, left_indices)
            if node.right:
                right_indices = indices[X[indices, node.feature_idx] > node.threshold]
                traverse(node.right, right_indices)
        
        # Устанавливаем такой же порядок признаков, как в обучающей выборке 
        X = X[self.features].to_numpy()
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_classes))
        traverse(self.root, np.arange(n_samples))
        return probs[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) > .5).astype(int)
        
    def _entropy(self, sample: np.ndarray) -> float:
        eps = 1e-12
        _, counts = np.unique(sample, return_counts=True)
        probabilities = np.clip(counts / np.sum(counts), eps, 1)
        return -(probabilities @ np.log2(probabilities))

    def _gini(self, sample: np.ndarray) -> float:
        _, counts = np.unique(sample, return_counts=True)
        probabilities = np.square(counts / np.sum(counts))
        return 1 - np.sum(probabilities)
