from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, _lit_doc, regexp_replace, col, udf
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Imputer


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Soen 471 Final Project") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def one_hot_encoding(input_df, cat_var, uniq_id):
    vals = input_df.select(cat_var).drop_duplicates().dropna().rdd.map(lambda x: x[0]).collect()
    enc_df = input_df.groupBy(uniq_id) \
        .pivot(cat_var, values=vals) \
        .agg(f.lit(1))
    if cat_var == "AccountabilityandAssistanceLevel" or cat_var == "District_AccountabilityandAssistanceLevel":
        for val in vals:
            new_column_name = cat_var + "_" + val
            enc_df = enc_df.withColumnRenamed(val, new_column_name)
    enc_df = enc_df.toDF(*(c.replace(" ", "_") for c in enc_df.columns))
    enc_df = enc_df.toDF(*(c.replace(",", "") for c in enc_df.columns))
    enc_df = enc_df.drop(vals[0]).na.fill(0)
    enc_df.printSchema()
    input_df = input_df.drop(cat_var).join(enc_df, on=uniq_id, how="inner")
    return input_df


def generate_test_train_split(output_df, output_variable, bin):
    var_df = output_df.select("Town", "SchoolCode", output_variable).dropna()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    ov_df = output_df.select("Town", output_variable).dropna()
    ov_df = ov_df.groupby("Town").agg({output_variable: 'mean'})
    output_variable = ov_df.drop("Town").schema.names[0]
    ov_pd = ov_df.filter(ov_df[output_variable] > 0).toPandas()
    ov_cat = output_variable + "_cat"
    ov_pd[ov_cat] = pd.cut(ov_pd[output_variable], bins=bin, labels=range(1, len(bin)))
    ax = ov_pd[ov_cat].hist()
    ax.set_title(f"Stratified {output_variable}")
    ax.set_xlabel("Strata")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    fig.savefig(f"../figures/{output_variable}_stratification.png")
    fig.clf()
    for train_index, test_index in split.split(ov_pd, ov_pd[ov_cat]):
        train = ov_pd.loc[train_index]
        test = ov_pd.loc[test_index]
    spark = init_spark()
    train_df = spark.createDataFrame(train).drop(ov_cat).drop(ov_cat.replace("_cat", ""))
    train_df = train_df.join(var_df, on="Town", how="inner").drop("Town").dropna()
    test_df = spark.createDataFrame(test).drop(ov_cat).drop(ov_cat.replace("_cat", ""))
    test_df = test_df.join(var_df, on="Town", how="inner").drop("Town")
    return [test_df, train_df]


def split_training_test_data_and_save(features_df, ov_test_split, ov_train_split, folder_name):
    test_df = ov_test_split.join(features_df, on="SchoolCode", how="inner")
    test_df.write.parquet(f"../{folder_name}/test_data.parquet")
    test_df.toPandas().to_csv(f"../{folder_name}/test_data.csv")
    #print(test_df.count())
    #test_df.printSchema()
    train_df = ov_train_split.join(features_df, on="SchoolCode", how="inner")
    #print(train_df.count())
    train_df.toPandas().to_csv(f"../{folder_name}/training_data.csv")
    train_df.write.parquet(f"../{folder_name}/training_data.parquet")
    train_df.toPandas().to_csv(f"../{folder_name}/training_data.csv")
    #train_df.printSchema()
    #print(test_df.select("Town").intersect(train_df.select("Town")).count())


def clean_training_and_test_data_and_save(folder_name):
    spark = init_spark()
    test_df = spark.read.parquet(f"../{folder_name}/test_data.parquet")
    imp_test_df = impute_df(test_df)
    imp_test_df.toPandas().to_csv(f"../{folder_name}/imputed_test_data.csv")
    imp_test_df.write.parquet(f"../{folder_name}/imputed_test_data.parquet")
    train_df = spark.read.parquet(f"../{folder_name}/training_data.parquet")
    imp_train_df = impute_df(train_df)
    imp_train_df.toPandas().to_csv(f"../{folder_name}/imputed_training_data.csv")
    imp_train_df.write.parquet(f"../{folder_name}/imputed_training_data.parquet")


def impute_df(df):
    spark = init_spark()
    df_columns = df.schema.names
    with open("../do_not_impute.txt") as f:
        do_not_impute = f.readlines()
        for feature in do_not_impute:
            feature = feature.replace("\n", "")
            df_columns.remove(feature)
    imp_columns = [f"{x}_imputed" for x in df_columns]
    impute_cols = Imputer(inputCols=df_columns, outputCols=imp_columns)
    imp_df = impute_cols.fit(df).transform(df)
    for col_name in df_columns:
        imp_df = imp_df.drop(col_name)
    return imp_df




