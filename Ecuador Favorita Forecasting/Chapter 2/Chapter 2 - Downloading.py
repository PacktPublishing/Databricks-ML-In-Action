# Databricks notebook source
# MAGIC %run ./../setup

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data",'/dbfs/FileStore/LakehouseInAction/')

# COMMAND ----------


