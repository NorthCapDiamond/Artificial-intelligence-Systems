from math_lib import *
from dima_learns_ml import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time


def lab_solver(data_frame):
	df = data_frame
	#Lets see our data frame
	print("Show Data Frame:\n")
	print(df)
	print()
	print("Data Shape", df.shape)
	print("=======================================")
	print()
	print("Check our data for NaN values:")
	print(df.isna().sum())
	print()

	df_columns = df.columns
	Y = df[["Wine"]]
	X = df.drop(["Wine"], axis=1)

	for i in df_columns:
		print("=======================================")
		if(df[i].dtype == "float" or df[i].dtype == "int"):
			print(i, "-- type of data is", df[i].dtypes)
			print("Amount of elements is", len(df[i]))
			print("Min value is", df[i].min(), "Max value is", df[i].max())
			print("Mean value is", df[i].mean(), "\nMode value is:\n", df[i].mode())
			print("Quantiles are:")
			print(df[i].quantile([0.25, 0.5, 0.75]))
			print("Unbiased variance value is", df[i].var())
			print("Standard deviation value is", df[i].std())
			print("Coefficient of variation is", cv(df[i]))
			print("\n")
		else:
			print(i, "-- type of data is", df[i].dtypes)
			print("Amount of elements is", len(df[i]))
			print("Mode value is:\n", df[i].mode())
			print("\n")
	print("=======================================")
	print("=======================================")



	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=df.Wine)
	scaler = MinMaxScaler().fit(X_train)


	X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
	start = time.time()
	pred = KNeighborsClassifier().fit(X_train, y_train.values.ravel()).predict(X_test)
	end = time.time()
	print("Sklearn")
	print("F1 score:", f1_score(y_test, pred, average="micro"))
	print(confusion_matrix(y_test, pred))
	print("Time:", end-start)

	print("=======================================")

	X_train, X_test, y_train, y_test = my_train_test_split(X, Y)
	n, k = kfold_knn(X_train, y_train, "Wine", [1,2], [5,3,7,9,11,13,15])
	start = time.time()
	model = KNNClassifier(k=k, n=n)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)
	end = time.time()
	print("My KNN")
	print("F1 score:", my_f1_score(y_test, pred))
	print("Cov", my_confusion_matrix(y_test, pred))
	print("Time:", end-start)
	print("=======================================")


df = pd.read_csv("WineDataset.csv")
lab_solver(df)
