import pandas as pd 
import numpy as np
from DecisionTree import DecisionTree
import random


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

	classes = list(set(y_test.tolist()))
	num_classes = len(classes)
	confusion_matrix = np.zeros((num_classes, num_classes))
	for true_label, predicted_label in zip(y_test, pred):
		true_label_index = classes.index(true_label)
		predicted_label_index = classes.index(predicted_label)
		confusion_matrix[true_label_index][predicted_label_index] += 1

	return confusion_matrix


def kfold_dtree(X, Y, label, max_depth, min_samples_split):
	df = pd.concat([X.copy(),Y.copy()], axis=1)
	df = df.rename(columns={label: "Label"}) 
	df = df.reindex(np.random.permutation(df.index)) 
	df = df.reset_index(drop=True)


	size_of_fold = X.shape[0]//10
	folds = []
	tmp = 0
	for i in range(10):
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
	best_l_rate = -1
	best_f1 = 0
	curr_f1 = 0


	for n in max_depth:
		for k in min_samples_split:
			curr_f1 = 0
			for i, val in enumerate(train):
				X_train, y_train, X_test, y_test = train[i][0].drop(["Label"], axis=1), train[i][0][["Label"]], test[i][0].drop(["Label"], axis=1), test[i][0][["Label"]]
				model = DecisionTree(max_depth=n, min_samples_split=k)
				model.fit(X_train.to_numpy(), y_train.to_numpy().flatten())
				curr_f1 += my_f1_score(pd.DataFrame(y_test), model.predict(X_test.to_numpy()))
			if(curr_f1/len(train) >= best_f1):
				best_n = n 
				best_l_rate = k
				best_f1 = curr_f1/len(train)


	print("Best depth", best_n)
	print("Best min split", best_l_rate)
	return [best_n, best_l_rate]