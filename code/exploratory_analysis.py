import matplotlib.pyplot as plt
import math
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from tabulate import tabulate


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Soen 471 Final Project") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def generate_histograms_for_output_variable(output_df):
    with open("../data/output_cleaned.txt") as f:
        output_variables = f.readlines()
        output_variables.remove("SchoolCode\n")
        output_variables.remove("Town\n")
        for output_variable in output_variables:
            output_variable = output_variable.replace("\n", "")
            output_np = output_df.select(output_variable).dropna().toPandas()
            n = output_np.count()
            nos_of_intervals = math.ceil(math.sqrt(n))
            plt.hist(output_np, bins=nos_of_intervals)
            plt.xlabel(output_variable)
            plt.ylabel("Number of Schools")
            title = output_variable + " vs Number of Schools"
            plt.title(title)
            plt.savefig(f"../figures/{output_variable}_vs_Number_of_Schools.png")
            plt.clf()
    return None


def generate_histograms_for_output_variable_avg_by_town(output_df):
    with open("../data/output_cleaned.txt") as f:
        output_variables = f.readlines()
        output_variables.remove("SchoolCode\n")
        output_variables.remove("Town\n")
        for output_variable in output_variables:
            output_variable = output_variable.replace("\n", "")
            output_variable_df = output_df.select("SchoolCode", "Town", output_variable).dropna()
            output_variable_pd = output_variable_df.groupBy("Town").agg({output_variable: "mean"}).drop("Town")\
                .toPandas()
            n = output_variable_pd.count()
            nos_of_intervals = math.ceil(math.sqrt(n))
            plt.hist(output_variable_pd, bins=nos_of_intervals)
            output_variable_string = output_variable + "(avg)"
            plt.xlabel(output_variable_string)
            plt.ylabel("Number of Towns")
            title = output_variable_string + " vs Number of Towns"
            plt.title(title)
            plt.savefig(f"../figures/{output_variable_string}_vs_Number_of_Towns.png")
            plt.clf()


def generate_box_plots_for_census_input_parameters(parameter_df):
    parameters = parameter_df.schema.names
    parameters.remove("PLACE")
    for param in parameters:
        parameter_np = parameter_df.select(param).dropna().toPandas()
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(parameter_np)
        title = param
        plt.title(title)
        plt.savefig(f"../figures/{param}.png")
        plt.clf()


def generate_box_plots_for_school_input_parameters(parameter_df):
    parameters = parameter_df.schema.names
    with open("../data/school_categorical_characteristics.txt") as f:
        categorical_parameters = f.readlines()
        for cat_param in categorical_parameters:
            cat_param = cat_param.replace("\n", "")
            print(cat_param)
            parameters.remove(cat_param)
    for param in parameters:
        parameter_np = parameter_df.select(param).dropna().toPandas()
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(parameter_np)
        title = param
        plt.title(title)
        plt.savefig(f"../figures/{param}.png")
        plt.clf()


def generate_schools_per_town():
    spark = init_spark()
    school_df = spark.read.parquet("../data/school_characteristics.parquet")
    with open("../figures/Count_Of_School_Town.txt", "x") as f:
        f.write(tabulate(school_df.groupBy("Town").count().orderBy("count").toPandas(), headers=["Town", "Count of Schools"]))




"""Taken from Sharareh's code in Exploratory Analysis Jupyter Notebook"""

# def generate_correlation_scores(census_df):
