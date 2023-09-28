# Databricks notebook source
# MAGIC %md
# MAGIC #Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

raw_data_path = volume_data_path + 'raw_data'
dbutils.fs.mkdirs(raw_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Download data
# MAGIC
# MAGIC Using the opendatasets library, connect to Kaggle and download the favorita data

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data",raw_data_path)
