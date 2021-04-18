import regression_utilities
from sklearn.tree import DecisionTreeRegressor


def baseline_decision_tree_regression():
    clf = DecisionTreeRegressor(random_state=0)
    regression_utilities.regress(clf, "dt_baseline", "x_train", "x_test")


def hyper_parameter_tuning_decision_tree_regression():
    clf = DecisionTreeRegressor(random_state=0)
    #regression_utilities.hyper_parameter_tuning(clf, ""