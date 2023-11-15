## Лабораторная работа 3. Деревья решений

### Введение

1. Для студентов с четным порядковым номером в группе – датасет с [классификацией грибов](https://archive.ics.uci.edu/ml/datasets/Mushroom), а нечетным – [датасет с данными про оценки студентов инженерного и педагогического факультетов](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation) (для данного датасета нужно ввести метрику: студент успешный/неуспешный на основании грейда)
2. Отобрать **случайным** образом sqrt(n) признаков
3. Реализовать без использования сторонних библиотек построение дерева решений (numpy и pandas использовать можно, использовать списки для реализации  дерева - нельзя)
4. Провести оценку реализованного алгоритма с использованием Accuracy, precision и recall
5. Построить AUC-ROC и AUC-PR (в пунктах 4 и 5 использовать библиотеки нельзя)

### Описание метода

Дерево решений — метод представления решающих правил в определенной иерархии, включающей в себя элементы двух типов — узлов (node) и листьев (leaf). Узлы включают в себя решающие правила и производят проверку примеров на соответствие выбранного атрибута обучающего множества.Oct 12, 2020

### Псевдокод метода

```css
import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def check_leaf_node(self):
        return self.value is not None

    def check_not_leaf_node(self):
        return self.value is None

    def set_feature(self, feature):
        self.feature = feature

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_left_node(self, left_ptr):
        self.left = left_ptr

    def set_right_node(self, right_ptr):
        self.right = right_ptr

    def set_value(self, value):
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def set_min_samples(self, min_samples_split):
        self.min_samples_split = min_samples_split

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

    def set_n_features(self, n_features):
        self.n_features = n_features

    def _split(self, X_column, split_thresh):
        return np.argwhere(X_column <= split_thresh).flatten(), np.argwhere(X_column > split_thresh).flatten()

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check if we Need to STOP!!!
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # else:
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain - best_gain > 0:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)
        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def fit(self, X, y):
        if(self.n_features):
            self.n_features = min(X.shape[1], self.n_features)
        else:
            self.n_features = X.shape[1]

        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.check_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


```

### Результаты выполнения

Best f1 is 1.0
Best depth 400
Best min split 5
Confusion matrix:
 [[1. 1. 0. 0. 0. 0. 0. 0.]
 [0. 6. 1. 0. 1. 0. 0. 0.]
 [0. 3. 3. 0. 0. 2. 0. 0.]
 [1. 2. 0. 0. 1. 1. 0. 0.]
 [0. 0. 0. 2. 0. 0. 1. 0.]
 [0. 1. 0. 0. 3. 1. 1. 1.]
 [0. 0. 0. 1. 0. 0. 2. 2.]
 [0. 0. 0. 0. 0. 0. 4. 2.]]
Accuracy: 0.875
Precision: 0.5
Recall: 1.0
False positive rate: 0.14285714285714285
Specificity: 0.8571428571428571

[AUC-ROC](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/blob/main/lab6/AUCS/Снимок%20экрана%202023-11-15%20в%2009.08.00.png) and [AUC-PR](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/blob/main/lab6/AUCS/Снимок%20экрана%202023-11-15%20в%2009.08.17.png)

### Примеры использования метода

Чаще всего метод дерева решений используют в сложных, но поддающихся классификации задачах принятия решений, когда перед нами есть несколько альтернативных "решений" (проектов, выходов, стратегий), каждое из которых в зависимости от наших действий или действий других лиц (а также глобальных сил, вроде рынка, природы и т.п.) может давать разные последствия (результаты).
