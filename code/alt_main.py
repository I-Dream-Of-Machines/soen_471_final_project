import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt


def generate_xy(output_variable, type):
    spark = init_spark()
    column_name = output_variable + "_imputed"
    df = spark.read.parquet(f"../data/{output_variable}/imputed_{type}_data.parquet")
    pd = df.toPandas().sort_values(by="SchoolCode")
    x_pd = pd.drop(columns=["SchoolCode", "Town", column_name])
    features = x_pd.columns
    x = x_pd.to_numpy()
    y = pd[column_name].to_numpy()
    return x, y, features


x_train, y_train, f = generate_xy("MCAS_10thGrade_Math_CPI", "training")
x_test, y_test, f = generate_xy("MCAS_10thGrade_Math_CPI", "test")

print(f)
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

ctr = DecisionTreeRegressor(random_state=0)
ctr = ctr.fit(x_train, y_train)
predictions = ctr.predict(x_test)
r = r2_score(predictions, y_test)
mae = mean_absolute_error(predictions, y_test)
mse = mean_squared_error(predictions, y_test)
print(ctr.get_depth())
print(r)
print(mae)
print(mse)

ctr = DecisionTreeRegressor(random_state=0, max_depth=5)
ctr = ctr.fit(x_train, y_train)
predictions = ctr.predict(x_test)
r = ctr.score(x_test, y_test)
mae = mean_absolute_error(predictions, y_test)
mse = mean_squared_error(predictions, y_test)
print(ctr.get_depth())
print(r)
print(mae)
print(mse)

ctr = DecisionTreeRegressor(random_state=0, max_depth=5)
ctr = ctr.fit(x_train, y_train)
predictions = ctr.predict(x_test)
r = ctr.score(x_test, y_test)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=600)
tree.plot_tree(ctr,
               feature_names=f,
               filled=True)
fig.savefig('10_math_cpi.png')
mae = mean_absolute_error(predictions, y_test)
mse = mean_squared_error(predictions, y_test)
tree_features = tree.plot_tree(ctr, feature_names=f)
for feature in tree_features:
    print(feature)
print(ctr.get_depth())
print(r)
print(mae)
print(mse)
