import exploratory_analysis
import preprocess_raw_data
import generate_test_training_sets
import utilities
import numpy as np


def main():
    #preprocess_raw_data.preprocess_all_data()
    #exploratory_analysis.generate_histograms_for_output_variable()
    #exploratory_analysis.generate_histograms_for_output_variable_avg_by_town()
    #generate_test_training_sets.combine_census_features()
    #generate_test_training_sets.combine_all_features()
    generate_test_training_sets.generate_test_training_sets()







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
