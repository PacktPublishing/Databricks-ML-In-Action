# Databricks notebook source
# MAGIC %run ../global-setup $project_name=favorita_forecasting

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data",'/dbfs/FileStore/LakehouseInAction/')

# COMMAND ----------


