import pandas as pd
import numpy as np
import utilities
from sklearn import metrics
import csv
import pickle as pk
import os

"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code 
Modified by Nadia 
"""


def print_save_metrics(y_pred, technique, ov):
    y_test = pd.read_csv(f"../data/test_training_data/{ov}/y_test.csv", sep=":")
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    with open(f"../results/{technique}/{ov}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["r2_score", "Mean Absolute Error (MAE)", 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)'])
        writer.writerow([r2_score, mae, mse, rmse])
    print("r2_score:" + ov, r2_score)
    print('Mean Absolute Error (MAE):' + ov, mae)
    print('Mean Squared Error (MSE):' + ov, mse)
    print('Root Mean Squared Error (RMSE):' + ov, rmse)


"""
Based on Shahreh's function random_forest()
Refactored by Nadia to ensure usability for both Random Forests and Decision Trees
"""


def regress(regressor, technique, ov):
    x_train = pd.read_csv(f"../data/test_training_data/{ov}/x_train.csv", sep=":")
    y_train = pd.read_csv(f"../data/test_training_data/{ov}/y_train.csv", sep=":")
    x_test = pd.read_csv(f"../data/test_training_data/{ov}/x_test.csv", sep=":")
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    with open(f'../models/{technique}/{ov}.pkl', 'wb') as model_file:
        pk.dump(regressor, model_file)
    with open(f'../results/{technique}/{ov}.pkl', 'wb') as prediction_file:
        pk.dump(y_pred, prediction_file)
    return regressor, y_pred


"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code 
Modified to save feature importance
"""


def feature_importance(regressor_feature_importance, technique, ov):
    x_test_columns = pd.read_csv(f"../data/test_training_data/{ov}/x_train.csv", ":").columns
    column_map = utilities.init_column_maps()
    important_features = sorted(list(zip(regressor_feature_importance, x_test_columns)), key=lambda x: x[0],
                                reverse=True)[:10]
    readable_important_features = []
    for feature in important_features:
        feature = (feature[0], feature[1].replace('_imputed', ''))
        column_map_tuple = list(filter(lambda column_name: feature[1] == column_name[0], column_map))[0]

        readable_important_features.append((feature[0], column_map_tuple[2]))
    file = open(f'../results/{technique}/{ov}.csv', 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(readable_important_features)
    print(readable_important_features)
    return None
