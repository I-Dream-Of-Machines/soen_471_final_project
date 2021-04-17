import pandas as pd
import numpy as np
from sklearn import metrics
import csv

"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code
"""


def train_test_split(ov):
    training = pd.read_csv('../data/test_training_data/' + ov + '/final_training_data.csv', sep=':')
    y_train = training.filter(regex=ov)
    x_train = training.drop(y_train, axis=1)
    test = pd.read_csv('../data/test_training_data/' + ov + '/final_test_data.csv', sep=':')
    y_test = test.filter(regex=ov)
    x_test = test.drop(y_test, axis=1)
    return x_train, x_test, y_train, y_test


"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code
"""


def print_save_metrics(y_test, y_pred, ov, metric_file_path):
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    with open(metric_file_path + ov + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["r2_score", "Mean Absolute Error (MAE)", 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)'])
        writer.writerow([r2_score, mae, mse, rmse])
    print("r2_score:" + ov, r2_score)
    print('Mean Absolute Error (MAE):' + ov, mae)
    print('Mean Squared Error (MSE):' + ov, mse)
    print('Root Mean Squared Error (RMSE):' + ov, rmse)


