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
# MAGIC SELECT COUNT(DISTINCT store_nbr) AS num_stores, COUNT(DISTINCT type) AS num_store_types
# MAGIC FROM lakehouse_in_action.favorita_forecasting.favorita_stores
# MAGIC WHERE state='Guayas'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT type, COUNT(*) AS num_stores
# MAGIC FROM lakehouse_in_action.favorita_forecasting.favorita_stores
# MAGIC WHERE state='Guayas'
# MAGIC GROUP BY all
# MAGIC ORDER BY

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT type, COUNT(*) AS num_stores
# MAGIC FROM lakehouse_in_action.favorita_forecasting.favorita_stores
# MAGIC WHERE state = 'Guayas'
# MAGIC GROUP BY type
# MAGIC ORDER BY num_stores DESC

# COMMAND ----------

from pyspark.sql.functions import *
df = spark.table("train_set")
df = df.withColumn("transaction_date", to_date("date"))
display(df)

# COMMAND ----------

from databricks import automl
summary = automl.regress(df, target_col="sales", timeout_minutes=30)

# COMMAND ----------


