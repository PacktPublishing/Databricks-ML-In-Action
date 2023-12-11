# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 4: Exploring and Cleaning Toward a Silver Layer
# MAGIC
# MAGIC ## Favorita Sales - Exploring Favorita Sales Data
# MAGIC
# MAGIC [Kaggle link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) and [autoML documentation](https://docs.databricks.com/en/machine-learning/automl/train-ml-model-automl-api.html)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------

from pyspark.sql.functions import *
df = spark.table("train_set")
df = df.withColumn("transaction_date", to_date("date"))
display(df)

# COMMAND ----------

from databricks import automl
summary = automl.regress(df, target_col="sales", timeout_minutes=30)

# COMMAND ----------


