# spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, _lit_doc, regexp_replace, col, udf
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType, IntegerType

# python imports


raw_social_characteristics_file_path = "../data/raw_data/ACSDP5Y2017.DP02-2021-03-23T230834.csv"
percentage_features_social_characteristics_file_path = "../parameters/social_characteristics/percentage_social_characteristics.txt"



def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Soen 471 Final Project") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def preprocess_school_characteristics(raw_data_file_path, parameter_file_path):
    df = clean_column_names(raw_data_file_path)
    parameter_list = generate_select_parameter(parameter_file_path)
    df = df.select(parameter_list)
    with open("../data/school_double_characteristics.txt") as parameters:
        parameter_list = parameters.readlines()
        parameter_list.remove("SchoolCode\n")
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(DoubleType()))
    with open("../data/school_integer_characteristics.txt") as parameters:
        parameter_list = parameters.readlines()
        parameter_list.remove("SchoolCode\n")
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(IntegerType()))
    return df


def preprocess_output_features(raw_data_file_path, output_file_path):
    spark = init_spark()
    df = clean_column_names(raw_data_file_path)
    parameter_list = generate_select_parameter(output_file_path)
    df = df.select(parameter_list)
    with open("../data/output_double_characteristics.txt") as parameters:
        parameter_list = parameters.readlines()
        parameter_list.remove("SchoolCode\n")
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(DoubleType()))
    with open("../data/output_integer_characteristics.txt") as parameters:
        parameter_list = parameters.readlines()
        parameter_list.remove("SchoolCode\n")
        for parameter in parameter_list:
            parameter = parameter.replace("\n", "")
            df = df.withColumn(parameter, col(parameter).cast(IntegerType()))
    return df


def clean_column_names(raw_data_file_path):
    spark = init_spark()
    df = spark.read.csv(raw_data_file_path, header=True, encoding="utf-8")
    clean_df = df.toDF(*(c.replace(".", "_") for c in df.columns))
    clean_df = clean_df.toDF(*(c.replace("`", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("\"", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("\"", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace(",", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace(";", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("{", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("}", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("(", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace(")", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("\t", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("\n", "") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace("!", "_") for c in clean_df.columns))
    clean_df = clean_df.toDF(*(c.replace(" ", "") for c in clean_df.columns))
    old_column_names = clean_df.schema.names
    new_column_names = [x.strip() for x in old_column_names]
    for i in range(0, len(old_column_names)):
        clean_df = clean_df.withColumnRenamed(old_column_names[i], str(new_column_names[i]))
    return clean_df

def generate_column_mapping():
    with open()


def generate_select_parameter(parameter_file):
    with open(parameter_file, encoding="utf-8") as parameters:
        parameter_list = parameters.readlines()
        parameter_list = list(map(lambda x: x.strip(), parameter_list))
        parameter_list = list(filter(lambda x: x != "", parameter_list))
        parameter_list = list(map(lambda x: x.replace("`", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace(".", "_"), parameter_list))
        parameter_list = list(map(lambda x: x.replace("\"", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace(",", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace(";", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace("{", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace("}", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace("(", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace(")", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace("\t", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace("\n", ""), parameter_list))
        parameter_list = list(map(lambda x: x.replace("!", "_"), parameter_list))
        parameter_list = list(map(lambda x: x.replace(" ", ""), parameter_list))
        with open(parameter_file.replace(".txt", "_cleaned.txt"), "x") as f:
            for parameter in parameter_list:
                f.write(parameter + "\n")
    return parameter_list


def shift_town_labels(input_df, value_label):
    df_values = input_df.filter(input_df["Label"] == value_label)
    df_towns = input_df.filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Estimate") \
        .filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Margin of Error") \
        .filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Percent") \
        .filter(input_df["Label"] != "\xa0\xa0\xa0\xa0Percent Margin of Error").select("Label")
    w = Window().orderBy("Label")
    df_values = df_values.withColumn("row_num", row_number().over(w))
    df_towns = df_towns.withColumn("row_num", row_number().over(w))
    df_towns = df_towns.withColumnRenamed("Label", "PLACE")
    df = df_towns.join(df_values, df_values.row_num == df_towns.row_num).drop("row_num").drop(
        "Label")
    df = df.filter(df["PLACE"] != 'undefined')
    return df


def remove_percentage(col_val):
    if '%' in col_val:
        col_val = col_val.replace('%', "")
        return float(col_val)
    else:
        return col_val


def preprocess_census_data(raw_data_file_path, percentage_parameter_file=None, estimate_parameter_file=None):
    df = clean_column_names(raw_data_file_path)
    if percentage_parameter_file is not None:
        parameter_list = generate_select_parameter(percentage_parameter_file)
        df_percentage = df.select(parameter_list)
        df_percentage = shift_town_labels(df_percentage, "\xa0\xa0\xa0\xa0Percent")
        cast_to_float = udf(lambda x: remove_percentage(x), StringType())
        with open(percentage_parameter_file.replace(".txt", "_cleaned.txt")) as parameters:
            parameter_list = parameters.readlines()
            parameter_list.remove("Label\n")
            for parameter in parameter_list:
                parameter = parameter.replace("\n", "")
                df_percentage = df_percentage.withColumn(parameter, cast_to_float(col(parameter)).cast(DoubleType()))
    if estimate_parameter_file is not None:
        parameter_list = generate_select_parameter(estimate_parameter_file)
        df_estimate = df.select(parameter_list)
        df_estimate.printSchema()
        df_estimate = shift_town_labels(df_estimate, "\xa0\xa0\xa0\xa0Estimate")
        df_estimate.printSchema()
        remove_comma = udf(lambda x: x.replace(",", ""), StringType())
        with open(estimate_parameter_file.replace(".txt", "_cleaned.txt")) as parameters:
            parameter_list = parameters.readlines()
            parameter_list.remove("Label\n")
            for parameter in parameter_list:
                parameter = parameter.replace("\n", "")
                df_estimate = df_estimate.withColumn(parameter, remove_comma(col(parameter)).cast(IntegerType()))
    if percentage_parameter_file is None:
        df = df_estimate
    elif estimate_parameter_file is None:
        df = df_percentage
    else:
        df_estimate = df_estimate.withColumnRenamed("PLACE", "ESTIMATE_PLACE")
        df = df_percentage.join(df_estimate, df_estimate.ESTIMATE_PLACE == df_percentage.PLACE)
        df = df.drop("ESTIMATE_PLACE")
    df = df.withColumn("PLACE", regexp_replace("PLACE", " CDP, Massachusetts", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " city, Massachusetts", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " Center", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " Corner", ""))
    df = df.withColumn("PLACE", regexp_replace("PLACE", " Town", ""))
    return df


def preprocess_and_save_demographic_characteristics():
    demographic_characteristics_file_path = "../data/ACSDP5Y2017.DP05-2021-03-24T134209.csv"
    estimate_demographic_characteristics_file_path = "../data/estimate_demographic_characteristics.txt"
    percentage_demographic_characteristics_file_path = "../data/percentage_demographic_characteristics.txt"
    demographic_df = preprocess_census_data(demographic_characteristics_file_path,
                                            percentage_demographic_characteristics_file_path,
                                            estimate_demographic_characteristics_file_path)
    demographic_df.write.parquet("../data/demographic_characteristics.parquet")


def preprocess_and_save_social_characteristics():
    social_characteristics_file_path = "../data/ACSDP5Y2017.DP02-2021-03-23T230834.csv"
    percentage_social_characteristics_file_path = "../data/percentage_social_characteristics.txt"
    social_df = preprocess_census_data(social_characteristics_file_path,
                                       percentage_parameter_file=percentage_social_characteristics_file_path)
    social_df.write.parquet("../data/social_characteristics.parquet")


def preprocess_and_save_economic_characteristics():
    economic_characteristics_file_path = "../data/ACSDP5Y2017.DP03-2021-03-24T112004.csv"
    estimate_economic_characteristics_file_path = "../data/estimate_economic_characteristics.txt"
    percentage_economic_characteristics_file_path = "../data/percentage_economic_characteristics.txt"
    economic_df = preprocess_census_data(economic_characteristics_file_path,
                                         percentage_economic_characteristics_file_path,
                                         estimate_economic_characteristics_file_path)
    economic_df.write.parquet("../data/economic_characteristics.parquet")


def preprocess_and_save_housing_characteristics():
    housing_characteristics_file_path = "../data/ACSDP5Y2017.DP04-2021-03-24T123953.csv"
    percentage_housing_characteristics_file_path = "../data/percentage_housing_characteristics.txt"
    housing_df = preprocess_census_data(housing_characteristics_file_path,
                                        percentage_parameter_file=percentage_housing_characteristics_file_path)
    housing_df.write.parquet("../data/housing_characteristics.parquet")


def preprocess_and_save_school_characteristics():
    school_data_file_path = "../data/MA_Public_Schools_2017.csv"
    school_characteristics_file_path = "../data/school_characteristics.txt"
    school_df = preprocess_school_characteristics(school_data_file_path, school_characteristics_file_path)
    school_df.write.parquet("../data/school_characteristics.parquet")


def preprocess_and_save_output_variables():
    school_data_file_path = "../data/MA_Public_Schools_2017.csv"
    output_variables_file_path = "../data/output.txt"
    output_df = preprocess_output_features(school_data_file_path, output_variables_file_path)
    output_df.printSchema()
    output_df.write.parquet("../data/output_variables.parquet")


def combine_census_features(social_characteristics_df, economic_characteristics_df,
                            housing_characteristics_df, demographic_characteristics_df, ):
    social_characteristics_df = social_characteristics_df.withColumnRenamed("PLACE", "SC_PLACE")
    economic_characteristics_df = economic_characteristics_df.withColumnRenamed("PLACE", "EC_PLACE")
    housing_characteristics_df = housing_characteristics_df.withColumnRenamed("PLACE", "HC_PLACE")
    demographic_characteristics_df = demographic_characteristics_df.withColumnRenamed("PLACE", "DC_PLACE")
    df = social_characteristics_df.join(economic_characteristics_df,
                                        social_characteristics_df.SC_PLACE == economic_characteristics_df.EC_PLACE)
    df = df.join(housing_characteristics_df, df.SC_PLACE == housing_characteristics_df.HC_PLACE)
    df = df.join(demographic_characteristics_df, df.SC_PLACE == demographic_characteristics_df.DC_PLACE)
    df = df.withColumnRenamed("SC_PLACE", "PLACE").drop("EC_PLACE").drop("HC_PLACE").drop("DC_PLACE")
    return df
