# Databricks notebook source
# MAGIC %md
# MAGIC # ASL Fingerspelling
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
# MAGIC
# MAGIC ##Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

raw_data_path = volume_data_path + 'raw_data'
dbutils.fs.mkdirs(raw_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Download data
# MAGIC
# MAGIC Using the opendatasets library, connect to Kaggle and download the asl data

# COMMAND ----------

import opendatasets as od

od.download("https://www.kaggle.com/competitions/asl-fingerspelling/data", raw_data_path)
