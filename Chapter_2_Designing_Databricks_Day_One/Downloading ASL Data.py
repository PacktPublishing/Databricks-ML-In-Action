# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 2: Designing Databricks Day One
# MAGIC
# MAGIC ## ASL Fingerspelling - Downloading ASL Data
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
# MAGIC

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Set the data path
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
