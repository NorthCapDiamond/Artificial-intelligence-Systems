# Отчёт по модулю №1

## Введение:

### Описание целей проекта и его значимости.

Целью моего проекта стала разработка системы поддержки принятия решения на основе запросов F1 на основе онтологии, разработанной в [лабораторной работе 2](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab2), которая, строилась на основе базы знаний из [лабораторной работы 1](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab1).

## Анализ требований:

### Определение основных требований к системе поддержки принятия решений.

- Получение информации о конкретном Racer, Racing Team, Event, Race и связях между ними (и их instances)
- Простота в использовании (система предназначена для людей с разными знаниями в области F1 races)
- Множественность вывода (в зависимости от введенных пользователем данных, система должна менять реакцию)
- Надежность. Необходимо обработать некорректно введенные данные пользователя

### Выявление требований к базе знаний и онтологии для представления знаний.
- Создание Classes: Racer, Racing Team, Event, Race
- Заполнение классов корректными instances
- Создание связей между ними (Object properties)
- Создание правил для управления данными 
- создание тестовых запросов

## Изучение основных концепций и инструментов:

### Обзор основных концепций баз знаний и онтологий.

Базы знаний - это системы, которые хранят информацию и знания, которые могут быть использованы для решения задач. Они обычно содержат факты, правила, процедуры и другие элементы, необходимые для решения задач. Базы знаний используются в различных областях, таких как медицина, финансы, право и т.д.

Онтологии - это формальные модели, которые описывают концептуальные категории и отношения между ними. Они используются для описания знаний в языке, который может быть понят людьми и машинами. Онтологии используются для создания более точных и эффективных систем поиска, классификации и анализа данных.

Основными принципами баз знаний являются:

- Декомпозиция знаний на факты, правила и процедуры.
- Построение моделей знаний на основе логических и математических принципов.

Основными принципами онтологий являются:

- Описание концептуальных категорий и отношений между ними.
- Использование формальных языков для описания знаний.
- Создание стандартных моделей для обмена и интеграции данных.


### Изучение Prolog и его возможностей для разработки систем искусственного интеллекта.

1) Принципы и концепции
   - Логическое программирование (факты, правила, поиск решений на основе утверждений)
   - Сопоставление значений и термов для поиска решения задачи
   - Оптимизация (ускорение работы программы путем отбрасывания лишних вариантов; бывают красные и зелёные в зависимости от того, влияют они на логику программы или нет)
   - Рекурсия (поддержка рекурсивных вызовов)
2) Синтаксис
   - Предикаты (описание отношений)
   - Переменные (искомые значения)
   - Константы (числа, атомы, строки)
   - Операторы (арифметические, логические, сравнения)
3) Решение задач искусственного интеллекта
   - Формализация знаний и автоматическое принятие решений
   - Доказательство теорем
   - Анализ естественного языка
4) Функциональные возможности
   - Работа с БЗ (запросы, поиск решений)
   - Логический вывод (вывод новых фактов на основе имеющихся)
   - Обработка естественного языка (полезен для систем обработки текстов и диалогов)
5) Преимущества
   - Декларативность
   - Удобство
   - Решение задач ичкусственного интеллекта
6) Недостатки
   - Ограничен на больших наборах данных
   - Сложность в отладке
7) Актуальность
   - Язык считается мертвым

### Ознакомление с инструментами и библиотеками, подходящими для работы с базами знаний и онтологиями на Prolog.
Примеры: OWL, PL/SQL, SQLAlchemy, TensorFlow, MongoDB, ...
## Реализация системы искусственного интеллекта на Prolog:

Примеры:
- Машинное обучение
- Обработка естественного языка
- Составление поддержки принятия решения

Реализация же мой системы доступна [lab3](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab3).


Примеры запросов в БЗ (это тесты) и онтологию продемонстрированы в соответствующих отчетах к [лабораторной №1](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab1) и [лабораторной №2](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab2).

Сравнение БЗ и Онтологии  их:

|     Что       |  Реализация в БЗ   |                   Реализация в онтологии                    |
|:-------------:|:------------------:|:-----------------------------------------------------------:|
|     Факт      |      Переменная    |                    Количество элементов                     |
|     Связь     |      Правило       |                     Наличие properties                      |
|   Рекурсия    |      Итерации      |          Reasoner обеспечивает установление связей          |


### Оценка и интерпретация результатов:

prolog:

is_winner_of_race("BAHRAIN GP", "Max Verstappen").

is_winner_of_race("BAHRAIN GP", "Dmitry Drobysh").

is_second_in_race("BAHRAIN GP", "Max Verstappen").

is_second_in_race("BAHRAIN GP", "Sergio Pérez").

is_third_in_race("BAHRAIN GP", "Fernando Alonso").

is_third_in_race("BAHRAIN GP", "Max Verstappen").


is_race_winner_in_team("BAHRAIN GP", "Red Bull").

is_race_second_in_team("BAHRAIN GP", "Red Bull").

is_race_second_in_team("BAHRAIN GP", "Ferrari").

is_race_third_in_team("BAHRAIN GP", "McLauren").

is_race_third_in_team("BAHRAIN GP", "Ferrari").

is_race_third_in_team("BAHRAIN GP", "Alpine").

is_race_third_in_team("BAHRAIN GP", "Aston Martin").



is_team_on_podium("BAHRAIN GP", "Red Bull").

is_team_on_podium("BAHRAIN GP", "Alpine").


protege: 


% Rule0 Return All parameters
Event

Race

Racer

Racing_Team




% Rule1 Show Racers From Team

Be_In_Team value Red_Bull




% Rule2 Return The Winner/Second/Third Of The Race
Win_The_Race value BAHRAIN_GP

Be_The_Second_In_Race value AUSTRALIAN_GP

Be_The_Third_In_Race value SAUDI_ARABIAN_GP




% Rule3 Return event that was in race
Event_In_Race value BAHRAIN_GP




% Rule4 Return Team Of A Winner/Second/Third
inverse(Be_In_Team) some (Win_The_Race value BAHRAIN_GP)

inverse(Be_In_Team) some (Be_The_Second_In_Race value AUSTRALIAN_GP)

inverse(Be_In_Team) some (Be_The_Third_In_Race value SAUDI_ARABIAN_GP)




% Rule5 Return Teams on podium
inverse(Be_In_Team) some (Be_On_Podium value AUSTRALIAN_GP)




% Rule6 see the racers that DNF
DNF value BAHRAIN_GP




% Rule7 see teams, where one or both pilots DNF
inverse(Be_In_Team) some (DNF value BAHRAIN_GP)




% Rule8 See The Partner of a Racer
Be_A_Partner value Max_Verstappen




% Rule9 See team if both drivers are on podium
((inverse (Be_In_Team) some (Win_The_Race value BAHRAIN_GP)) and (inverse (Be_In_Team) some (Be_The_Second_In_Race value BAHRAIN_GP))) or
((inverse (Be_In_Team) some (Win_The_Race value BAHRAIN_GP)) and (inverse (Be_In_Team) some (Be_The_Third_In_Race value BAHRAIN_GP))) or
((inverse (Be_In_Team) some (Be_The_Second_In_Race value BAHRAIN_GP)) and (inverse (Be_In_Team) some (Be_The_Second_In_Race value BAHRAIN_GP)))




% Rule10 See team if one driver on podium and another one DNF
((inverse (Be_In_Team) some (Be_On_Podium value SAUDI_ARABIAN_GP)) and (inverse (Be_In_Team) some (DNF value SAUDI_ARABIAN_GP)))


SPRQL: 

// info param + 

// show racers from team param +

// i like to see the team that dnf  +

// i like to see the racer that dnf +

// i hate to see the team that dnf  +

// i hate to see the racer that dnf +

// i like to see the winner of param +

// i hate to see the winner of param +

// i like param racers 

// i hate param racers 



### Оценка соответствия системы поставленным требованиям и достижению целей проекта.

!!!Это решать моему практику!!!

### Интерпретация результатов и описание дальнейших возможностей развития и улучшения системы.

Полученные результаты можно интерпретировать как ИИ-помощника, действующего на основе онтологии, способной советовать пользователю команд и гонщиков из F1

## Заключение:
В ходе выполнения ЛР1,2,3 я ознакомился с языком Prolog, научился реализовывать на нем базы знаний. Кроме того, я научился работать в Protege и составлять онтологии. Я также написал собственную систему принятия решений на основе Python 3 и онтологий в формате rdf.
Система уже может использоваться.









# Модуль 2. Шаблон отчёта

## Лабораторная работа 1. Метод линейной регрессии

### Введение

Целью этой лабораторной работы является реализация линейной регрессии с использованием метода наименьших квадратов.

- Студенты с четным порядковым номером в группе должны использовать набор данных о [жилье в Калифорнии] (https://developers.google.com/machine-learning/crash-course/california-housing-data-description?hl=ru). [Скачать тут](https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv).
- Получите и визуализируйте статистику по датасету (включая количество, среднее значение, стандартное отклонение, минимум, максимум и различные квантили).
- Проведите предварительную обработку данных, включая обработку отсутствующих значений, кодирование категориальных признаков и нормировка.
- Разделите данные на обучающий и тестовый наборы данных.
- Реализуйте линейную регрессию с использованием метода наименьших квадратов без использования сторонних библиотек, кроме NumPy и Pandas. Использовать минимизацию суммы квадратов разностей между фактическими и предсказанными значениями для нахождения оптимальных коэффициентов.
- Постройте **три модели** с различными наборами признаков.
- Для каждой модели проведите оценку производительности, используя метрику коэффициент детерминации, чтобы измерить, насколько хорошо модель соответствует данным.
- Сравните результаты трех моделей и сделайте выводы о том, какие признаки работают лучше всего для каждой модели.
- Бонусное задание
    - Ввести синтетический признак при построении модели

### Описание метода

Метод наименьших квадратов (МНК) — математический метод, применяемый для решения различных задач, основанный на минимизации суммы квадратов отклонений некоторых функций от экспериментальных входных данных.

### Псевдокод метода

```css
def LinearRegressionScalarCoefficients(X, Y):
	ones = pd.DataFrame([1]*X.shape[0])
	X = pd.concat([ones, X], axis=1)
	array = []


	matrix = []
	for i in X.columns:
		tmp_array = []
		for j in X.columns:
			tmp_array.append((X[i]*X[j]).sum())
		tmp_array.append((Y[Y.columns[0]]*X[i]).sum())
		matrix.append(tmp_array)
	return gauss(matrix)

def LinearRegressionScalar(x_train, y_train, x_test, copy=True):

	if copy:
		X_train = x_train.copy()
		X_test = x_test.copy()
		Y_train = y_train.copy()
	else:
		X_train = x_train
		X_test = x_test
		Y_train = y_train


	X_train = MinMaxScaler(X_train)
	X_test = MinMaxScaler(X_test)
	answer = []
	coeffs = LinearRegressionScalarCoefficients(X_train, Y_train)


	for i in range(X_test.shape[0]):
		tmp = 0
		row = X_test.iloc[i].tolist()
		row.insert(0, 1)
		for j in range(len(coeffs)):
			tmp += row[j]*coeffs[j]
		answer.append(tmp)
	return pd.DataFrame(answer).to_numpy()





def normal_equation(X, Y):
	return np.dot( np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

def predict(X, norm):
	return np.dot(X, norm)

def LinearRegressionMatrix(x_train, y_train, x_test, copy=True):
	if copy:
		X_train = x_train.copy()
		X_test = x_test.copy()
		Y_train = y_train.copy()
	else:
		X_train = x_train
		X_test = x_test
		Y_train = y_train

	X_train = MinMaxScaler(X_train)
	X_test = MinMaxScaler(X_test)
	X_train = X_train.to_numpy()
	X_test = X_test.to_numpy()
	Y_train = Y_train.to_numpy()

	one1 = np.ones((len(X_train),1))
	one2 = np.ones((len(X_test),1))
	X_train = np.append(one1, X_train, axis=1)
	X_test = np.append(one2, X_test, axis=1)

	return(predict(X_test, normal_equation(X_train, Y_train)))

```

### Результаты выполнения

Correlated features are: households, total_bedrooms, total_rooms => PCA used

Sklearn
r2_score: 0.6273355474455504
Time: 0.014391183853149414
=======================================
Scalar
r2_score: 0.26107710904267056
Time: 0.07764697074890137
=======================================
Matrix
r2_score: 0.6205272420946046
Time: 0.003862142562866211
=======================================

### Примеры использования метода

Если мы что-то знаем о распределении данных и хотим предсказать непрерывную величину, то можно использовать модель LinReg. Если данные распределены не совсем линейно, то можно попробовать их обработать и получить нужный результат 

## Лабораторная работа 2. Метод k-ближайших соседей (k-NN)

### Введение

Проведите предварительную обработку данных, включая обработку отсутствующих значений, кодирование категориальных признаков и масштабирование.
Реализуйте метод k-ближайших соседей ****без использования сторонних библиотек, кроме NumPy и Pandas.
Постройте две модели k-NN с различными наборами признаков:
Модель 1: Признаки случайно отбираются .
Модель 2: Фиксированный набор признаков, который выбирается заранее.
Для каждой модели проведите оценку на тестовом наборе данных при разных значениях k. Выберите несколько различных значений k, например, k=3, k=5, k=10, и т. д. Постройте матрицу ошибок.

### Описание метода

Метод k-ближайших соседей (k-nearest neighbors) – это простой алгоритм машинного обучения с учителем, который можно использовать для решения задач классификации и регрессии. Он прост в реализации и понимании, но имеет существенный недостаток – значительное замедление работы, когда объем данных растет.

### Псевдокод метода

```css
def minkowski_distance(x, y, n):
	return np.power((np.sum(abs(x - y)**n)), 1 / n)


class KNNClassifier:
    def __init__(self, k=3, n=2):
        if not(k % 2):
            pred("K must be non odd")
            exit(0)
        self.k = k
        self.n = n

    def fit(self, X, Y):
        self.X_train = MyMinMaxScaler(X).to_numpy()
        self.Y_train = Y.to_numpy()

    def predict(self, X):
        X = MyMinMaxScaler(X).to_numpy()

        def inner_pred(x):
            distances = [minkowski_distance(x, y, self.n)
                         for y in self.X_train]

            k_indexes = np.argsort(distances)[:self.k]
            k_labels = [self.Y_train[i] for i in k_indexes]

            k_labels_tuples = [tuple(row) for row in k_labels]


            mode = Counter(k_labels_tuples).most_common()

            return mode[0][0][0]

        return [inner_pred(x) for x in X]

```

### Результаты выполнения

=======================================
Sklearn
F1 score: 0.9074074074074074
[[18  0  0]
 [ 1 16  4]
 [ 0  0 15]]
Time: 0.00279998779296875
=======================================
Best f1 is 1.0
Best n 2
Best k 15
My KNN
F1 score: 0.9245283018867925
Cov [[19.  1.  0.]
 [ 1. 18.  2.]
 [ 0.  0. 12.]]
Time: 0.02074599266052246
=======================================

### Примеры использования метода

almost everywhere.. but it is too slow

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


## Лабораторная работа 4. Логистическая регрессия

### Введение

1. Для студентов с четным порядковым номером в группе – датасет с [классификацией грибов](https://archive.ics.uci.edu/ml/datasets/Mushroom), а нечетным – [датасет с данными про оценки студентов инженерного и педагогического факультетов](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation) (для данного датасета нужно ввести метрику: студент успешный/неуспешный на основании грейда)
2. Отобрать **случайным** образом sqrt(n) признаков
3. Реализовать без использования сторонних библиотек построение дерева решений (numpy и pandas использовать можно, использовать списки для реализации  дерева - нельзя)
4. Провести оценку реализованного алгоритма с использованием Accuracy, precision и recall
5. Построить AUC-ROC и AUC-PR (в пунктах 4 и 5 использовать библиотеки нельзя)

## Лабораторная 7.  Логистическая регрессия

[логистическая регрессия.docx](https://prod-files-secure.s3.us-west-2.amazonaws.com/d5c92538-65d6-4833-ae99-f27868eeaf39/55221bbd-8a47-42f7-878b-a1608bdf4120/%D0%BB%D0%BE%D0%B3%D0%B8%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F.docx)

1. Выбор датасета:
- Датасет о пассажирах Титаника: [Titanic Dataset](https://www.kaggle.com/c/titanic)
- Датасет о диабете: [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- Загрузите выбранный датасет и выполните предварительную обработку данных.
- Получите и визуализируйте (графически) статистику по датасету (включая количество, среднее значение, стандартное отклонение, минимум, максимум и различные квантили).
- Разделите данные на обучающий и тестовый наборы в соотношении, которое вы считаете подходящим.
- Реализуйте логистическую регрессию "с нуля" без использования сторонних библиотек, кроме NumPy и Pandas. Ваша реализация логистической регрессии должна включать в себя:
    - Функцию для вычисления гипотезы (sigmoid function).
    - Функцию для вычисления функции потерь (log loss).
    - Метод обучения, который включает в себя градиентный спуск.
    - Возможность варьировать гиперпараметры, такие как коэффициент обучения (learning rate) и количество итераций.
1. Исследование гиперпараметров:
    - Проведите исследование влияния гиперпараметров на производительность модели. Варьируйте следующие гиперпараметры:
        - Коэффициент обучения (learning rate).
        - Количество итераций обучения.
        - Метод оптимизации (например, градиентный спуск или оптимизация Ньютона).
2. Оценка модели:
    - Для каждой комбинации гиперпараметров оцените производительность модели на тестовом наборе данных, используя метрики, такие как accuracy, precision, recall и F1-Score.

### Описание метода

Логистическая регрессия — это метод анализа данных, который использует математику для поиска взаимосвязей между двумя факторами данных. Затем эта взаимосвязь используется для прогнозирования значения одного из этих факторов на основе другого.

### Псевдокод метода

```css
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    def __init__(self, n_iter=1000, l_rate=0.001):
        self.n_iter = n_iter
        self.l_rate = l_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        N, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.n_iter):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)
            dw = (1 / N) * 2 * np.dot(X.T, predictions - y)
            db = (1 / N) * 2 * sum(predictions - y)

            self.weights = self.weights - self.l_rate * dw
            self.bias = self.bias - self.l_rate * db

    def predict(self, X, percentile=0.5):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_predictions)

        predict_class = [1 if y > percentile else 0 for y in y_predicted]
        return predict_class


```

### Результаты выполнения

Best f1 is 0.730263157894737
Best n 5000
Best k 0.0001
F1 score: 0.6782608695652174
Confusion matrix:
 [[156.   0.]
 [ 74.   0.]]
Accuracy: 0.6782608695652174
Precision: 1.0
Recall: 0.6782608695652174

### Примеры использования метода

Логистическая регрессия полезна для ситуаций, в которых вы хотите иметь возможность предсказать наличие или отсутствие характеристики или итога на основании значений набора переменных - предикторов.

## Сравнение методов

### Сравнительный анализ методов
По типу:
- Все с учителем!
По задаче:
- Logistic reg, Knn, DTrees -> Классификация
- LinReg - регрессия 

1. Логистическая регрессия (Logreg):
Преимущества:

Простота и быстрота обучения.
Эффективен, когда данные линейно разделимы.
Малое количество гиперпараметров.
Ограничения:

Неспособен улавливать сложные нелинейные взаимосвязи.
Чувствителен к выбросам.

2. Метод k-ближайших соседей (KNN):
Преимущества:
Простота реализации и понимания.
Способен обрабатывать нелинейные отношения.
Не требует обучения в процессе.

Ограничения:
Чувствителен к масштабированию признаков.
Затратен по вычислительным ресурсам на этапе предсказания.

3. Деревья решений (Decision Trees):
Преимущества:

Способны обрабатывать как линейные, так и нелинейные зависимости.
Могут визуализироваться для лучшего понимания решающего процесса.
Могут автоматически обрабатывать отсутствующие значения.
Ограничения:

Склонны к переобучению, особенно на шумных данных.
Неустойчивы к изменениям в данных.

4. Линейная регрессия (LinReg):
Преимущества:
Простота и интерпретируемость: Линейная регрессия - один из самых простых и легко интерпретируемых методов машинного обучения. Результаты модели могут быть объяснены с использованием коэффициентов, соответствующих каждому признаку.

Эффективность на линейных данных: Линейная регрессия работает эффективно, когда существует линейная зависимость между предикторами и целевой переменной. В таких случаях она может предоставить точные и стабильные прогнозы.

Отсутствие гиперпараметров: Обучение линейной регрессии не требует настройки сложных гиперпараметров, что делает его привлекательным для быстрого прототипирования и базового анализа данных.

Ограничения:
Линейная зависимость: Линейная регрессия не эффективна, если зависимость между предикторами и целевой переменной нелинейна.

Чувствительность к выбросам: Линейная регрессия подвержена влиянию выбросов в данных, что может существенно повлиять на коэффициенты модели.

Ограниченность в выражении сложных взаимосвязей: Не способна улавливать сложные нелинейные взаимосвязи и взаимодействия между признаками.




### Примеры лучшего использования каждого метода

Logreg:
Лучшее использование: Когда данные линейно разделимы, и важна интерпретируемость модели, например, в медицинских исследованиях для прогнозирования вероятности заболевания.

LinReg:
Примеры лучшего использования:
Прогнозирование цен на недвижимость: Когда предполагается, что цена недвижимости линейно зависит от факторов, таких как площадь, количество комнат и расстояние до центра города.
Экономические прогнозы: В задачах, где предполагается линейная зависимость между экономическими показателями, например, ВВП и уровнем инвестиций.
Прогнозирование объема продаж: Когда предполагается, что объем продаж линейно зависит от рекламных затрат и других маркетинговых факторов.
и тп.

KNN:
Лучшее использование: В задачах, где области принятия решений имеют сложную форму и нелинейные взаимосвязи, например, в задачах распознавания образов.

DTrees:
Лучшее использование: Когда в данных присутствуют сложные нелинейные зависимости и важна способность объяснить принятое решение. Например, в финансовых анализах для прогнозирования решений инвестиций.


## Заключение

В ходе выполнения ЛР 4-7 я освоил все основные принципы и методы работы с алгоритмами ML и необходимыми для их оценками методами. Более того, написал собственные реализации и сравнил с популярной библиотекой sklearn. 
## Приложения

[Код]


