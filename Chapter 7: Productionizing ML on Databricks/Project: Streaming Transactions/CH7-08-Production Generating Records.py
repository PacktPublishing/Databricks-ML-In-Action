# Databricks notebook source
# MAGIC %md Chapter 7: Productionizing ML on Databricks
# MAGIC
# MAGIC ## Production Generating Records

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

dbutils.widgets.text('raw_table_name','prod_transactions','Enter raw table name')
table_name = dbutils.widgets.get('raw_table_name')

# COMMAND ----------

destination_path = f"{volume_file_path}/{table_name}/data/"
dbutils.fs.mkdirs(destination_path)
temp_path = "{}/temp".format(volume_file_path)
sleepIntervalSeconds = 1

# COMMAND ----------

from mlia_utils.transactions_funcs import writeJsonFile

import time

while(1==1):
  writeJsonFile(spark,destination_path,temp_path)
  time.sleep(sleepIntervalSeconds)

# COMMAND ----------

# df = spark.read.format("json").load(destination_path)
# display(df)
