# Databricks notebook source
# MAGIC %sql
# MAGIC USE DATABASE lakehouse_in_action

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data","/dbfs/FileStore/LakehouseInAction/")

# COMMAND ----------


