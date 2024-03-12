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

CustomerID_vars = {"min": 1234, "max": 1260}
Product_vars = {"A": {"min": 1000, "max": 25001, "mean": 15520, "alpha": 4, "beta": 10},
                "B": {"min": 1000, "max": 5501, "mean": 35520, "alpha": 10, "beta": 4},
                "C": {"min": 10000, "max": 40001, "mean": 30520, "alpha": 3, "beta": 10}}

destination_path = f"{volume_file_path}/{table_name}/data/"
dbutils.fs.mkdirs(destination_path)
temp_path = "{}/temp".format(volume_file_path)
sleepIntervalSeconds = 1

# COMMAND ----------

from mlia_utils.transactions_funcs import writeJsonFile

import time

while(1==1):
  writeJsonFile(spark,destination_path,temp_path,Product_vars,CustomerID_vars)
  time.sleep(sleepIntervalSeconds)
