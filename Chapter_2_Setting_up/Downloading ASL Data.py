# Databricks notebook source
# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/asl-fingerspelling/data", "/dbfs/FileStore/LakehouseInAction/", force=True)

# COMMAND ----------

dbutils.fs.mv('dbfs:/FileStore/LakehouseInAction/asl-fingerspelling','s3://one-env/lakehouse_ml_in_action/asl-fingerspelling', recurse=True)

# COMMAND ----------


