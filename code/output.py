from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import pandas as pd
import pre_processing

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Soen 471 Final Project") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def write_to_csv():
    spark = init_spark()
    demographic_df = spark.read.parquet("../data/demographic_characteristics.parquet")
    social_df = spark.read.parquet("../data/social_characteristics.parquet")
    economic_df = spark.read.parquet("../data/economic_characteristics.parquet")
    housing_df = spark.read.parquet("../data/housing_characteristics.parquet")
    school_df = spark.read.parquet("../data/school_characteristics.parquet")
    df = pre_processing.combine_census_features(social_df, economic_df, housing_df, demographic_df)
    df.toPandas().to_csv("../data/census_data.csv")

