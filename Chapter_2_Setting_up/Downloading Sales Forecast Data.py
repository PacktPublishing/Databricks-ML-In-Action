# Databricks notebook source
# MAGIC %md
# MAGIC #Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md
# MAGIC #Download data
# MAGIC
# MAGIC Using the opendatasets library, connect to Kaggle and download the favorita data

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data","/dbfs/FileStore/LakehouseInAction/")

# COMMAND ----------

dbutils.fs.mv('dbfs:/FileStore/LakehouseInAction/store-sales-time-series-forecasting/',cloud_storage_path, recurse=True)

# COMMAND ----------

# MAGIC %fs ls s3://one-env/lakehouse_ml_in_action/favorita_forecasting/

# COMMAND ----------


