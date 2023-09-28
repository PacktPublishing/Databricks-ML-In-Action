# Databricks notebook source
# MAGIC %run ../global-setup $project_name=parkinsons-freezing_gait_prediction

# COMMAND ----------

dbutils.fs.mkdirs(volume_data_path + 'raw_data')

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data",volume_data_path + 'raw_data',force=True)

# COMMAND ----------


