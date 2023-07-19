# Databricks notebook source
# MAGIC %md
# MAGIC ## Parkinson's FOG
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/)

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=parkinsons-freezing_gait_prediction $catalog=lakehouse_in_action

# COMMAND ----------

import pandas as pd
import pyspark.pandas as ps
from  pyspark.sql.functions import input_file_name, split


# COMMAND ----------

display(dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transforming CSV to Delta

# COMMAND ----------

# DBTITLE 1,tDCS FOG metadata
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_tdcsfog_metadata")
display(df)

# COMMAND ----------

# DBTITLE 1,DeFOG metadata
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_defog_metadata")
display(df)

# COMMAND ----------

# DBTITLE 1,Daily DeFOG metadata
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/daily_metadata.csv', sep=',', decimal='.')
df.rename({'Beginning of recording [00:00-23:59]': 'RecordingStartTime'}, axis=1, inplace=True)
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_defog_daily_metadata")
display(df)

# COMMAND ----------

# DBTITLE 1,Subjects
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/subjects.csv', sep=',', decimal='.')
df['Visit'] = df['Visit'].astype('Int64') #consistency of data types between tables
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_subjects")
display(df)

# COMMAND ----------

# DBTITLE 1,Events
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/events.csv', sep=',', decimal='.')
df['Kinetic'] = df['Kinetic'].astype('Int64')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_fog_events")
display(df)

# COMMAND ----------

# DBTITLE 1,Tasks
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/tasks.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_defog_tasks")
display(df)

# COMMAND ----------

# DBTITLE 1,Sample submission
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/sample_submission.csv', sep=',', decimal='.')
display(df)

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/defog'))

# COMMAND ----------

# DBTITLE 1,DeFOG training
df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/defog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_train_defog")

# COMMAND ----------

# DBTITLE 1,NoType training
df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/notype/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_train_notype")

# COMMAND ----------

# DBTITLE 1,tDCS FOG training
df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_train_tdcsfog")

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/'))

# COMMAND ----------

# DBTITLE 1,tDCS FOG testing
df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_test_tdcsfog")

# COMMAND ----------

# DBTITLE 1,DeFOG testing

df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/defog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_test_defog")

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/unlabeled/'))

# COMMAND ----------

# DBTITLE 1,Unlabeled
df = spark.read.format("parquet") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/unlabeled/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[5]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_unlabeled")

# COMMAND ----------


