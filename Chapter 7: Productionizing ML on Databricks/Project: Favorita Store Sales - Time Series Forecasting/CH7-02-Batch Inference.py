# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Productionizing ML on Databricks
# MAGIC
# MAGIC ## Favorita Forecasting -Registering the Model
# MAGIC
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct, col
mlflow.set_registry_uri("databricks-uc")
model_name = "store_sales_forecasting"

logged_model = f"models:/{model_name}@Champion"

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

df = sql("SELECT * FROM test_set")

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))
