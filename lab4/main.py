from math_lib import *
from dima_learns_ml import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("california_housing_train.csv")
#df = pd.read_csv("Student_Performance.csv")

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


#Change this to yours:

Y = df[["median_house_value"]]
df = df.drop("median_house_value", axis=1)
flag_v2 = False

# Y = df[["Performance Index"]]
# df = df.drop("Performance Index", axis=1)
# flag_v2 = True

df_columns = df.columns

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
df = pd.get_dummies(df)
print(df)
print("=======================================")


#If next conditions are not the same in your variant, comment it:
print("Now, I'm gonna check if there are correlated features:\n")
sns.heatmap(df.corr(), annot=True)
plt.show()
# I see 2 most correlated features: "households", "total_bedrooms"
correlated_features = ["households", "total_bedrooms", "total_rooms"]
print("Correlated features are:", ", ".join(correlated_features))
print("=======================================")

#We need to use PCA for this:

#Lets prepare the data:
mergred_features = df[correlated_features]
target_for_pca = Y["median_house_value"]

#delete features from df

df = df.drop(correlated_features, axis=1)
print("=======================================")
# using PCA:
pca_applied = PCA(mergred_features, n_components=1)

#Now we wanna recreate df:
temporary_df = pd.DataFrame(pca_applied, columns=['hosts'])
print(temporary_df)
df = pd.concat([df, temporary_df], axis=1)
print(df)

sns.heatmap(df.corr(), annot=True)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(df, Y)
pred = LinearRegression().fit(X_train, y_train).predict(X_test)
print("Sklearn")
print("r2_score:", r2_score(y_test, pred))

print("=======================================")

X_train, X_test, y_train, y_test = my_train_test_split(df, Y)
pred = LinearRegressionScalar(X_train, y_train, X_test)
print("Scalar")
print("r2_score:", my_r2_score(y_test, pred))
print("=======================================")

if not(flag_v2):
	X_train, X_test, y_train, y_test = my_train_test_split(df, Y)
	pred = LinearRegressionMatrix(X_train, y_train, X_test)
	print("Matrix")
	print("r2_score:", my_r2_score(y_test, pred))
	print("=======================================")
