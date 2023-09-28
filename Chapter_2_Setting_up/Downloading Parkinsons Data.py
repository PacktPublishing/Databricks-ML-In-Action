# Databricks notebook source
# MAGIC %md
# MAGIC #Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=parkinsons-freezing_gait_prediction

# COMMAND ----------

dbutils.fs.mkdirs(volume_data_path + 'raw_data')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Download data
# MAGIC
# MAGIC Using the opendatasets library, connect to Kaggle and download the Parkinson's FOG data

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data",volume_data_path + 'raw_data',force=True)

# COMMAND ----------


