import pandas as pd
import numpy as np
from sklearn import metrics
import csv

"""
Code Contribution (Shahrareh) - take from her Jupyter Notebook while combining code
Modified by Nadia  to save generated train_test_split
"""


def train_test_split(ov):
    training = pd.read_csv(f"../data/test_training_data/{ov}/final_training_data.csv", sep=':')
    test = pd.read_csv(f"../data/test_training_data/{ov}/final_test_data.csv", sep=':')
    y_train = training.filter(regex=ov)
    x_train = training.drop(y_train, axis=1)
    y_test = test.filter(regex=ov)
    x_test = test.drop(y_test, axis=1)
    y_train.write_csv(f"../data/test_training_data/{ov}/y_train.csv", sep=":", index=False)
    x_train.write_csv(f"../data/test_training_data/{ov}/x_train.csv", sep=":", index=False)
    x_test.write_csv(f"../data/test_training_data/{ov}/x_train.csv", sep=":", index=False)
    y_test.write_csv(f"../data/test_training_data/{ov}/y_train.csv", sep=":", index=False)

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


def feature_importance(important_list):
    lst_column_map = init_column_maps()
    lst_final = []
    # important_list=sorted(list(zip(regressor.feature_importances_,X_test.columns)),key =lambda x: x[0] ,reverse=True)[:10]
    for import_item in important_list:
        import_item_rem = import_item[1].replace('_imputed', '')
        for column_item in lst_column_map:
            if import_item_rem == column_item[0]:
                import_renamed = column_item[2]
                l = list(import_item)
                l[1] = import_renamed
                lst_final.append(tuple(l))
                break
    return lst_final



def random_forest():
    output_variable = ['School_Code', 'OP1', 'OP2', 'OP6', 'OP3', 'OP4', 'OP5', 'OP7', 'OP8', 'OP9', 'OP10', 'OP11',
                       'OP12', 'OP13', 'OP14']
    output_variable.remove('School_Code')
    for item in output_variable:
        X_train, X_test, y_train, y_test = train_test_split(item)
        regressor = RandomForestRegressor()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        with open('../data/random_forest/' + item + '.pkl', 'wb') as pickle_file:
            pk.dump(y_pred, pickle_file)

        important_list = sorted(list(zip(regressor.feature_importances_, X_test.columns)), key=lambda x: x[0],
                                reverse=True)[:10]
        import_lst = feature_importance(important_list)
        file = open('../data/random_forest/feature_importance_' + item + '.csv', 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(import_lst)
        print_save_metrics(y_test, y_pred, item)


