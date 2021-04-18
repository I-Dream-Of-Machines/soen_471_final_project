import pandas as pd
import numpy as np
import utilities
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import csv
import pickle as pk


def x_split(ov):
    training = pd.read_csv(f"../data/test_training_data/{ov}/final_num_training_data.csv", sep=':')
    test = pd.read_csv(f"../data/test_training_data/{ov}/final_num_test_data.csv", sep=':')
    x_train = training.drop(ov, axis=1).drop(["School_Code"], axis=1)
    x_test = test.drop(ov, axis=1).drop(["School_Code"], axis=1)
    x_train.to_csv(f"../data/test_training_data/{ov}/x_num_train.csv", sep=":", index=False)
    x_test.to_csv(f"../data/test_training_data/{ov}/x_num_test.csv", sep=":", index=False)
    return x_test, x_train

def pca(ov):
    print("Hello")
    #return x_pca_train, x_pca_test

def generate_scree_plt(ov, pca):
    print("Hello")


def perform_pca():
    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for output_variable in output_variables:
            output_variable = output_variable.replace("\n", "")
            x_num_train, x_num_test = x_split(output_variable) #generates x_num_final_train, x_num_final_test
            #generate_heat_map(output_variable, x_num_train)
            pca()
            #generate_scree_plt(ov)
            #generate_heat_map
