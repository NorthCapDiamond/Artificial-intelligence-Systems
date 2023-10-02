from math_lib import *
from collections import Counter


def MyMinMaxScaler(A):
    for i in A.columns:
        maxi = A[i].max()
        mini = A[i].min()
        A[i] = (A[i] - mini) / (maxi - mini)
    return A


def my_train_test_split(X, Y, test_size=0.3):
    X_columns = X.columns
    Y_columns = Y.columns
    X = X.to_numpy()
    Y = Y.to_numpy()
    idxes = np.array(range(len(X)))

    test_size = round(test_size * len(X))

    random.shuffle(idxes)

    train_idx = idxes[test_size:len(X)]
    test_idx = idxes[0:test_size]

    return(pd.DataFrame(X[train_idx, :], columns=X_columns), pd.DataFrame(X[test_idx, :], columns=X_columns), pd.DataFrame(Y[train_idx, :], columns=Y_columns), pd.DataFrame(Y[test_idx, :], columns=Y_columns))


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



def my_f1_score(y_test, pred):
	#Classes = [TP, FP, TN, FN]
	y_test = np.array(y_test).flatten()
	pred = np.array(pred)
	TP = 0
	FP = 0
	TN = 0
	FN = 0


	classes = list(set(y_test.tolist()))
	classes_stats = [[0, 0, 0, 0]]*len(classes)
	for i, cur_class in enumerate(classes):
		for idx, (el1, el2) in enumerate(zip(y_test, pred)):
			if (el1==el2==cur_class):
				TP+=1
				classes_stats[i][0]+=1
			if (el2==cur_class and el1!=el2):
				FP+=1
				classes_stats[i][1]+=1
			if (el1==el2 and el1!=cur_class):
				classes_stats[i][2]+=1
				TN+=1
			if (el1!=el2 and el1!=cur_class):
				classes_stats[i][3]+=1
				FN+=1

	return TP/(TP + (1/(len(classes))) * (FP + FN))


def my_confusion_matrix(y_test, pred):
	#Classes = [TP, FP, TN, FN]
	y_test = np.array(y_test).flatten()
	pred = np.array(pred)
	TP = 0
	FP = 0
	TN = 0
	FN = 0


	classes = list(set(y_test.tolist()))
	classes_stats = [[0, 0, 0, 0]]*len(classes)
	for i, cur_class in enumerate(classes):
		for idx, (el1, el2) in enumerate(zip(y_test, pred)):
			if (el1==el2==cur_class):
				TP+=1
				classes_stats[i][0]+=1
			if (el2==cur_class and el1!=el2):
				FP+=1
				classes_stats[i][1]+=1
			if (el1==el2 and el1!=cur_class):
				classes_stats[i][2]+=1
				TN+=1
			if (el1!=el2 and el1!=cur_class):
				classes_stats[i][3]+=1
				FN+=1

	return classes_stats


def kfold_knn(X, Y, label, n_val, k_val):
	df = pd.concat([X.copy(),Y.copy()], axis=1)
	df = df.rename(columns={label: "Label"}) 
	df = df.reindex(np.random.permutation(df.index)) 
	df = df.reset_index(drop=True)


	size_of_fold = X.shape[0]//10
	folds = []
	tmp = 0
	for i in range(4):
		folds.append(pd.DataFrame(df.iloc[tmp:(tmp+size_of_fold)]))
		tmp += size_of_fold



	train = []
	test = [] 
	for i in range(len(folds)):
		tmp1 = []
		tmp2 = []
		for j in range(len(folds)):
			if i==j:
				tmp2.append(folds[i])
			else:
				if(len(tmp1)>0):
					tmp1[0] = pd.concat([tmp1[0], folds[i]])
				else:
					tmp1.append(folds[i])

		train.append(tmp1)
		test.append(tmp2)
	

	best_n = -1
	best_k = -1
	best_f1 = 0
	curr_f1 = 0


	for n in n_val:
		for k in k_val:
			curr_f1 = 0
			for i, val in enumerate(train):
				X_train, y_train, X_test, y_test = train[i][0].drop(["Label"], axis=1), train[i][0][["Label"]], test[i][0].drop(["Label"], axis=1), test[i][0][["Label"]]
				model = KNNClassifier(k=k, n=n)
				model.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))
				#print(X_test.shape, y_test.shape)
				curr_f1 += my_f1_score(pd.DataFrame(y_test), model.predict(pd.DataFrame(X_test)))
			if(curr_f1/len(train) >= best_f1):
				best_n = n 
				best_k = k
				best_f1 = curr_f1/len(train)


	print("Best f1 is",best_f1)
	print("Best n", best_n)
	print("Best k", best_k)
	return [best_n, best_k]
