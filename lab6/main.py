import pandas as pd
from dima_learns_ml import *
import seaborn as sns
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree
from LabelEncoding import *
from ROC import *

df = pd.read_csv("DATA.csv")
encoder = LabelEncoder()
y = df[["GRADE"]]
X = df.drop(["GRADE"], axis=1)
X["STUDENT ID"] = encoder.fit_transform(X["STUDENT ID"])
print(X)
print(y)


depth, mini = kfold_dtree(X, y, "GRADE", [100, 200, 300, 400], [1, 2, 3, 4, 5])

X_train, X_test, y_train, y_test = my_train_test_split(X, y)
X_train = MyMinMaxScaler(X_train).to_numpy()
X_test = MyMinMaxScaler(X_test).to_numpy()
y_train = y_train.to_numpy().flatten()
y_test = y_test.to_numpy().flatten()


model = DecisionTree(max_depth=depth, min_samples_split=mini)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


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

fpr, tpr = auc_roc(y_test, y_pred)
roc_auc = np.trapz(tpr, fpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

recall, precision = auc_pr(y_test, y_pred)
pr_auc = np.trapz(precision, recall)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = {:.2f})'.format(pr_auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
