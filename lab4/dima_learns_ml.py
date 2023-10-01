from math_lib import *
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


def MinMaxScaler(A):
	for i in A.columns:
		maxi = A[i].max()
		mini = A[i].min()
		A[i] = (A[i] - mini)/(maxi - mini)
	return A



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




def my_r2_score(Y_test, predictions):
    rss = np.sum(np.square((Y_test - predictions)))
    mean = np.mean(Y_test)
    sst = np.sum(np.square(Y_test-mean))
    r_square = 1 - (rss/sst)
    
    return r_square[0]
