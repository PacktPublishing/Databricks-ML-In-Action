# Databricks notebook source
# MAGIC %md 
# MAGIC Chapter 2 
# MAGIC
# MAGIC ##  Intel image multilabel classification - Dowloanding our images to the Volumes
# MAGIC We will download data from Kaggle Dataset: [Kaggle competition link](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data) 
# MAGIC It's a classical multiclass classification problem. 
# MAGIC
# MAGIC In the next chapter, we will ingest images into a delta table and prepare our labels for training. 

# COMMAND ----------

# First install all necessary libraties
!pip install kaggle

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

# DBTITLE 1,Add your credentials to connect to the Kaggle Account 
import os

os.environ[
    "kaggle_username"
] = dbutils.secrets.get("lakehouse-in-action", "kaggle_username")  # replace with your own credential here temporarily or set up a secret scope with your credential

os.environ[
    "kaggle_key"
] = dbutils.secrets.get("lakehouse-in-action", "kaggle_key")  # replace with your own credential here temporarily or set up a secret scope with your credential

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /local_disk0/
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle datasets download -d puneet6060/intel-image-classification

# COMMAND ----------

# DBTITLE 1,Unzip your data under Volumes 
!mkdir /Volumes/{catalog}/{database_name}/intel_image_clf/
!mkdir /Volumes/{catalog}/{database_name}/intel_image_clf/raw_images

# this can take up to 1h 
# or load a few examples to the UC on your own if the time is a constraint 
!unzip -n /local_disk0/intel-image-classification.zip -d /Volumes/{catalog}/{database_name}/intel_image_clf/raw_images 
