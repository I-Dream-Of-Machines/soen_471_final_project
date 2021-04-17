import utilities
import regression_utilities
from sklearn.tree import DecisionTreeRegressor


def regress_baseline_decision_tree():
    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        # output_variables = ["OP4"]
        for ov in output_variables:
            ov = ov.replace("\n", "")
            clf = DecisionTreeRegressor(random_state=0)
            regressor, y_pred = regression_utilities.regress(clf, "dt_baseline", ov)
            regression_utilities.print_save_metrics(y_pred, "dt_baseline", ov)
            regression_utilities.feature_importance(regressor.feature_importances_, "dt_baseline", ov)
            regression_utilities.print_save_regressor_params(regressor, "dt_baseline", ov)
