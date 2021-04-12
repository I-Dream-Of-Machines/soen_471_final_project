from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

import pandas as pd
import numpy as np


def main(preprocess=False, explore=False, encode=False, combine_features=False, split=False, clean=True, pca=False):
    spark = init_spark()

    # Preprocessing
    if preprocess:
        pre_processing.preprocess_and_save_demographic_characteristics()
        pre_processing.preprocess_and_save_social_characteristics()
        pre_processing.preprocess_and_save_economic_characteristics()
        pre_processing.preprocess_and_save_housing_characteristics()
        pre_processing.preprocess_and_save_school_characteristics()
        pre_processing.preprocess_and_save_output_variables()

    # Load preprocessed data
    demographic_df = spark.read.parquet("../data/demographic_characteristics.parquet")
    social_df = spark.read.parquet("../data/social_characteristics.parquet")
    economic_df = spark.read.parquet("../data/economic_characteristics.parquet")
    housing_df = spark.read.parquet("../data/housing_characteristics.parquet")
    school_df = spark.read.parquet("../data/school_characteristics.parquet")
    output_df = spark.read.parquet("../data/output_variables.parquet")

    # Exploratory Analysis
    if explore:
        exploratory_analysis.generate_box_plots_for_school_input_parameters(school_df)
        exploratory_analysis.generate_box_plots_for_census_input_parameters(demographic_df)
        exploratory_analysis.generate_box_plots_for_census_input_parameters(social_df)
        exploratory_analysis.generate_box_plots_for_census_input_parameters(economic_df)
        exploratory_analysis.generate_box_plots_for_census_input_parameters(housing_df)
        exploratory_analysis.generate_schools_per_town()
        exploratory_analysis.generate_histograms_for_output_variable(output_df)

    # Encoding
    if encode:
        with open("../data/school_categorical_characteristics.txt") as f:
            cat_params = f.readlines()
            cat_params.remove("SchoolCode\n")
            print(cat_params)
            for param in cat_params:
                param = param.replace("\n", "")
                school_df = encoding.one_hot_encoding(school_df, param, "SchoolCode")
        school_df.printSchema()
        school_df.write.parquet("../data/encoded_school_characteristics.parquet")
    else:
        school_df = spark.read.parquet("../data/encoded_school_characteristics.parquet")

    if combine_features:
        census_df = pre_processing.combine_census_features(social_df, economic_df, housing_df, demographic_df)
        features_df = school_df.join(census_df, census_df.PLACE == school_df.Town).drop("PLACE")
        features_df.write.parquet("../data/all_features.parquet")
    else:
        features_df = spark.read.parquet("../data/all_features.parquet")

    # Stratified Sampling
    if split:
        test_output, train_output = encoding.generate_test_train_split(output_df, "%Graduated",
                                                                       [0, 50, 60, 70, 85, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "graduation_rate")
        test_output, train_output = encoding.generate_test_train_split(output_df, "%AttendingCollege",
                                                                       [0, 40, 60, 80, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "college_attendance_rate")
        test_output, train_output = encoding.generate_test_train_split(output_df, "AverageSAT_Reading",
                                                                       [0, 400, 450, 500, 550, 600, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "average_sat_reading_score")
        test_output, train_output = encoding.generate_test_train_split(output_df, "AverageSAT_Writing",
                                                                       [0, 400, 450, 500, 550, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "average_sat_writing_score")
        test_output, train_output = encoding.generate_test_train_split(output_df, "AverageSAT_Math",
                                                                       [0, 400, 475, 550, 600, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "average_sat_math_score")
        test_output, train_output = encoding.generate_test_train_split(output_df, "%AP_Score3-5",
                                                                       [0, 20, 40, 60, 80, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "ap_3_5")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_3rdGrade_Math_CPI",
                                                                       [0, 70, 80, 90, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_3_math_cpi")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_5thGrade_Math_CPI",
                                                                       [0, 70, 80, 90, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_5_math_cpi")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_8thGrade_Math_CPI",
                                                                       [0, 60, 70, 80, 90, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_8_math_cpi")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_10thGrade_Math_CPI",
                                                                       [0, 70, 80, 90, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_10_math_cpi")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_3rdGrade_English_CPI",
                                                                       [0, 75, 85, 95, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_3_english_cpi")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_5thGrade_English_CPI",
                                                                       [0, 75, 80, 85, 90, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_5_english_cpi")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_8thGrade_English_CPI",
                                                                       [0, 85, 90, 95, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_8_english_cpi")
        test_output, train_output = encoding.generate_test_train_split(output_df, "MCAS_10thGrade_English_CPI",
                                                                       [0, 90, np.inf])
        encoding.split_training_test_data_and_save(features_df, test_output, train_output, "mcas_10_english_cpi")

    # Fill in Missing Values using Simple Imputer
    if clean:
        encoding.clean_training_and_test_data_and_save("graduation_rate")
        encoding.clean_training_and_test_data_and_save("college_attendance_rate")
        encoding.clean_training_and_test_data_and_save("average_sat_reading_score")
        encoding.clean_training_and_test_data_and_save("average_sat_writing_score")
        encoding.clean_training_and_test_data_and_save("average_sat_math_score")
        encoding.clean_training_and_test_data_and_save("ap_3_5")
        encoding.clean_training_and_test_data_and_save("mcas_3_math_cpi")
        encoding.clean_training_and_test_data_and_save("mcas_5_math_cpi")
        encoding.clean_training_and_test_data_and_save("mcas_8_math_cpi")
        encoding.clean_training_and_test_data_and_save("mcas_10_math_cpi")
        encoding.clean_training_and_test_data_and_save("mcas_3_english_cpi")
        encoding.clean_training_and_test_data_and_save("mcas_5_english_cpi")
        encoding.clean_training_and_test_data_and_save("mcas_8_english_cpi")
        encoding.clean_training_and_test_data_and_save("mcas_10_english_cpi")

    # Run Decision Tree

    # Grid Search
    # Run Decision Tree
    # PCA
    # Random Forest
    # Run Random Forest
    # Grid Search
    # PCA
    # Run Random Forest






