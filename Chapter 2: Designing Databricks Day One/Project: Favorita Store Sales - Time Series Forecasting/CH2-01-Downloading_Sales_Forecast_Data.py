# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 2: Designing Databricks Day One
# MAGIC
# MAGIC ## Favorita Forecasting - Downloading Sales Forecast Data
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Setup

# COMMAND ----------

# MAGIC
# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# DBTITLE 1,Set the data path

raw_data_path = volume_file_path + 'raw_data'

dbutils.fs.mkdirs(raw_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Download data
# MAGIC
# MAGIC Using the opendatasets library, connect to Kaggle and download the favorita data

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data",raw_data_path)


# COMMAND ----------

dbutils.fs.ls(raw_data_path + "/store-sales-time-series-forecasting/")

