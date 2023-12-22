# Databricks notebook source
# MAGIC %md 
# MAGIC Chapter 2 
# MAGIC
# MAGIC
# MAGIC We will download data from Kaggle Dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data
# MAGIC It's a classical multiclass classification problem. 
# MAGIC
# MAGIC We then will ingest images into a delta table and prepare our labels for training. 

# COMMAND ----------

# First install all necessary libraties
!pip install kaggle

# COMMAND ----------

# set your catalog name and the schema name 
# if you are using dbfs you dont need this 
dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "")

def use_and_create_db(catalog, schemaName):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{schemaName}` """)
  print(f"USE SCHEMA '{schema_name}'")
  spark.sql(f"USE {schemaName}")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
use_and_create_db(catalog_name, schema_name)

# COMMAND ----------

import os

os.environ[
    "kaggle_username"
] = dbutils.secrets.get("mlaction", "kaggle_name")  # replace with your own credential here temporarily or set up a secret scope with your credential

os.environ[
    "kaggle_key"
] = dbutils.secrets.get("mlaction", "kaggle_key")  # replace with your own credential here temporarily or set up a secret scope with your credential

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /local_disk0/
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle datasets download -d puneet6060/intel-image-classification

# COMMAND ----------

!mkdir /Volumes/{catalog_name}/{schema_name}/intel_image_clf/raw_images

# this can take up to 1h 
# or load a few examples to the UC on your own if the time is a constraint 
!unzip -n /local_disk0/intel-image-classification.zip -d /Volumes/ap/cv_uc/intel_image_clf/raw_images 
