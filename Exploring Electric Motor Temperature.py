# Databricks notebook source
pip install pip --upgrade

# COMMAND ----------

!pip install opendatasets
# pandas is already installed because we are using a ML runtime

# COMMAND ----------

import opendatasets as od
import pandas

od.download("https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature","/dbfs/FileStore/LakehouseInAction/")


# COMMAND ----------

# dbutils.fs.rm("dbfs:/FileStore/LakehouseInAction/electric-motor-temperature/electric-motor-temperature", recurse=True)

# COMMAND ----------

pip install bamboolib

# COMMAND ----------

import bamboolib as bam
bam

# COMMAND ----------


