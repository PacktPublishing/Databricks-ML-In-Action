# Databricks notebook source
username = spark.sql('select current_user() as user').collect()[0]['user']


# COMMAND ----------

import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_username'] = dbutils.secrets.get("lakehouse-in-action", "kaggle_username")

# os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_key'] = dbutils.secrets.get("lakehouse-in-action", "kaggle_key")

# COMMAND ----------

# DBTITLE 1,Create a database to store your tables
# MAGIC %sql
# MAGIC 
# MAGIC CREATE DATABASE hive_metastore.lakehouse_in_action

# COMMAND ----------

# MAGIC %md
# MAGIC Hopefully this will be in UC before publishing. But due to integrations NOT working with UC, for now, hive it is.
