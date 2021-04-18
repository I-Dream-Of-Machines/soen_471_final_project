import regression_utilities
from sklearn.tree import DecisionTreeRegressor


def baseline_decision_tree_regression():
    clf = DecisionTreeRegressor(random_state=0)
    regression_utilities.regress(clf, "dt_baseline", "x_train", "x_test", True)


def baseline_pca_decision_tree_regression():
    clf = DecisionTreeRegressor(random_state=0)
    regression_utilities.regress(clf, "dt_baseline_pca", "x_pca_train", "x_pca_test", False)


def hyper_parameter_tuning_decision_tree_regression():
    clf = DecisionTreeRegressor(random_state=0)
    #regression_utilities.hyper_parameter_tuning(clf, ""