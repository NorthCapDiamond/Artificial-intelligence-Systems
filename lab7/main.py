import pandas as pd
from dima_learns_ml import *
import seaborn as sns
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

df = pd.read_csv("diabetes.csv")
y = df[["Outcome"]]
X = df.drop(["Outcome"], axis=1)
print(X)
print(y)
sns.heatmap(df.corr(), annot=True)
plt.show()

print("Check our data for NaN values:")
print(df.isna().sum())
print()

n_iter, l_rate = kfold_logreg(X, y, "Outcome", [
                              100, 500, 800, 900, 1000, 1200, 1300, 1500, 2000, 3000, 5000, 10000], [0.01, 0.001, 0.0001, 0.00001])

X_train, X_test, y_train, y_test = my_train_test_split(X, y)
X_train = MyMinMaxScaler(X_train).to_numpy()
X_test = MyMinMaxScaler(X_test).to_numpy()
y_train = y_train.to_numpy().flatten()
y_test = y_test.to_numpy().flatten()

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print("F1 score:", my_f1_score(y_test, y_pred))


conf_matrix = my_confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_matrix)


TP = int(conf_matrix[0][0])
FP = int(conf_matrix[0][1])
FN = int(conf_matrix[1][0])
TN = int(conf_matrix[1][1])

if (TP + FP + FN + TN):
    print("Accuracy:", (TP + TN) / (TP + FP + FN + TN))
if (TP + FP):
    print("Precision:", (TP) / (TP + FP))
if (TP + FN):
    print("Recall:", (TP) / (TP + FN))
if (FP + TN):
    print("False positive rate:", FP / (FP + TN))
if(TN + FP):
    print("Specificity:", TN / (TN + FP))
