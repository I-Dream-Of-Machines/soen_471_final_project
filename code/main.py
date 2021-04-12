from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Soen 471 Final Project") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

spark = init_spark()
df = spark.read.csv("../data/raw_data/ACSDP5Y2017.DP03-2021-03-24T112004.csv", header=True, encoding="utf-8")
df.printSchema()


