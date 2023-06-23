# Databricks notebook source
# MAGIC %run ./setup

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data",'/dbfs/FileStore/LakehouseInAction/')

# COMMAND ----------


