# Databricks notebook source
# MAGIC %md
# MAGIC https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=parkinsons-freezing_gait_prediction

# COMMAND ----------

import pandas as pd
import pyspark.pandas as ps
from  pyspark.sql.functions import input_file_name, split


# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction')

# COMMAND ----------

# DBTITLE 1,Transforming CSV to Delta


# COMMAND ----------

# MAGIC %md
# MAGIC ### tdcsfog_metadata.csv 
# MAGIC Identifies each series in the tdcsfog dataset by a unique `Subject`, `Visit`, `Test`, `Medication` condition.
# MAGIC
# MAGIC * `Visit` Lab visits consist of a baseline assessment, two post-treatment assessments for different treatment stages, and one follow-up assessment.
# MAGIC * `Test` Which of three test types was performed, with 3 the most challenging.
# MAGIC * `Medication` Subjects may have been either off or on anti-parkinsonian medication during the recording.

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_tdcsfog_metadata")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### defog_metadata.csv 
# MAGIC Identifies each series in the defog dataset by a unique `Subject`, `Visit`, `Medication` condition.

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_defog_metadata")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### daily_metadata.csv 
# MAGIC Each series in the `daily` dataset is identified by the `Subject` id. This file also contains the time of day the recording began.

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/daily_metadata.csv', sep=',', decimal='.')
df.rename({'Beginning of recording [00:00-23:59]': 'RecordingStartTime'}, axis=1, inplace=True)
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_defog_daily_metadata")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### subjects.csv 
# MAGIC Metadata for each `Subject` in the study, including their `Age` and `Sex` as well as:
# MAGIC
# MAGIC * `Visit` Only available for subjects in the daily and defog datasets.
# MAGIC * `YearsSinceDx` Years since Parkinson's diagnosis.
# MAGIC * `UPDRSIIIOn/UPDRSIIIOff` Unified Parkinson's Disease Rating Scale score during on/off medication respectively.
# MAGIC * `NFOGQ` Self-report FoG questionnaire score. See: https://pubmed.ncbi.nlm.nih.gov/19660949/

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/subjects.csv', sep=',', decimal='.')
df['Visit'] = df['Visit'].astype('Int64') #consistency of data types between tables
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_subjects")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### events.csv 
# MAGIC Metadata for each FoG event in all data series. The event times agree with the labels in the data series.
# MAGIC
# MAGIC * `Id` The data series the event occured in.
# MAGIC * `Init` Time (s) the event began.
# MAGIC * `Completion` Time (s) the event ended.
# MAGIC * `Type` Whether StartHesitation, Turn, or Walking.
# MAGIC * `Kinetic` Whether the event was kinetic (1) and involved movement, or akinetic (0) and static.

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/events.csv', sep=',', decimal='.')
df['Kinetic'] = df['Kinetic'].astype('Int64')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_fog_events")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### tasks.csv 
# MAGIC Task metadata for series in the defog dataset. (Not relevant for the series in the `tdcsfog` or `daily` datasets.)
# MAGIC
# MAGIC * `Id` The data series where the task was measured.
# MAGIC * `Begin` Time (s) the task began.
# MAGIC * `End` Time (s) the task ended.
# MAGIC * `Task` One of seven tasks types in the DeFOG protocol, described on this page.

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/tasks.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("parkinsons_defog_tasks")
display(df)

# COMMAND ----------

# DBTITLE 1,Sample submission
df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/sample_submission.csv', sep=',', decimal='.')
display(df)

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/')

# COMMAND ----------

# MAGIC %md The DeFOG (defog) dataset, comprising data series collected in the subject's home, as subjects completed a FOG-provoking protocol

# COMMAND ----------

df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/defog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_train_defog")

# COMMAND ----------

display(df.select('id').distinct())

# COMMAND ----------

df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/notype/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_train_notype")

# COMMAND ----------

df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_train_tdcsfog")

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/')

# COMMAND ----------

df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_test_tdcsfog")

# COMMAND ----------


df = spark.read.format("csv") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/defog/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[6]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_test_defog")

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/unlabeled/')

# COMMAND ----------

df = spark.read.format("parquet") \
  .load(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/unlabeled/', sep=',', decimal='.',header=True,inferType=True) \
  .withColumn("file_id", input_file_name())

df = df.withColumn("id", split(split(df.file_id,'\.')[0],'/')[5]).drop('file_id')

display(df)
df.write.mode("overwrite").saveAsTable("parkinsons_unlabeled")

# COMMAND ----------


