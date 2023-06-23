# Databricks notebook source
# MAGIC %run ../global-setup $project_name=parkinsons-freezing_gait_prediction

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data",'/dbfs/FileStore/LakehouseInAction/')

# COMMAND ----------


