

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