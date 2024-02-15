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

# First install all necessary libraries
!pip install kaggle

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

# DBTITLE 1,Kaggle download using Databricks Secrets
# MAGIC %sh
# MAGIC cd /local_disk0/
# MAGIC kaggle datasets download -d puneet6060/intel-image-classification

# COMMAND ----------

# DBTITLE 1,Unzip your data under Volumes 
!mkdir {volume_file_path}/intel_image_clf/
!mkdir {volume_file_path}/intel_image_clf/raw_images

# this can take up to a few hours  
# or load a few examples to UC on your own if time is a constraint
!unzip -n /local_disk0/intel-image-classification.zip -d /Volumes/{catalog}/{database_name}/files/intel_image_clf/raw_images 
