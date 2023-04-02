# Databricks notebook source
# MAGIC %md
# MAGIC https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/

# COMMAND ----------

# MAGIC %run ./setup

# COMMAND ----------

import pandas as pd
import pyspark.pandas as ps

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction')

# COMMAND ----------

# DBTITLE 1,Transforming CSV to Delta
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/daily_metadata.csv', sep=',', decimal='.')
df.rename({'Beginning of recording [00:00-23:59]': 'RecordingStartTime'}, axis=1, inplace=True)
spark.createDataFrame(df).write.saveAsTable("parkinsons_defog_daily_metadata")

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.saveAsTable("parkinsons_defog_metadata")

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/events.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.saveAsTable("parkinsons_fog_events")

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/tasks.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.saveAsTable("parkinsons_defog_tasks")

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.saveAsTable("parkinsons_tdcsfog_metadata")

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/subjects.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.saveAsTable("parkinsons_subjects")

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/sample_submission.csv', sep=',', decimal='.')
display(df)

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/defog')

# COMMAND ----------

df = ps.read_csv(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/defog/', sep=',', decimal='.')
display(df)
df.to_table("parkinsons_defog_train")

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/')

# COMMAND ----------

df = ps.read_csv(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/notype/', sep=',', decimal='.')
display(df)
#df.to_table("parkinsons_notype_train")

# COMMAND ----------


