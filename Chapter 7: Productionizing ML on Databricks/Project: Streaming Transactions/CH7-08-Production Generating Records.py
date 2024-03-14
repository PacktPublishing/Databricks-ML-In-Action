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
raw_table_name = dbutils.widgets.get('raw_table_name')
dbutils.widgets.text('label_table_name','transaction_labels','Enter label table name')
label_table_name = dbutils.widgets.get('label_table_name')

# COMMAND ----------

record_path = f"{volume_file_path}/{raw_table_name}/data/"
dbutils.fs.mkdirs(record_path)
label_path = f"{volume_file_path}/{label_table_name}/data/"
dbutils.fs.mkdirs(label_path)
temp_path = "{}/temp".format(volume_file_path)
sleepIntervalSeconds = 1

# COMMAND ----------

from mlia_utils.transactions_funcs import writeJsonFile

import time

# while(1==1):
#   writeJsonFile(spark,destination_path,temp_path)
#   time.sleep(sleepIntervalSeconds)

t=1
while(t<3):
  writeJsonFile(spark,record_path,label_path,temp_path)
  time.sleep(sleepIntervalSeconds)
  t+=1

# COMMAND ----------

# df = spark.read.format("json").load(record_path)
# display(df)

# COMMAND ----------

# df = spark.read.format("json").load(label_path)
# display(df)
