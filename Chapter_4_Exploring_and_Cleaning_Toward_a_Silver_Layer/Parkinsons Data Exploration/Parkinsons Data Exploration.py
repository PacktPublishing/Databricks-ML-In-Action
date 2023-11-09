# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 4: Exploring and Cleaning Toward a Silver Layer
# MAGIC
# MAGIC ## Parkinson's FOG - Parkinsons Data Exploration
# MAGIC
# MAGIC [Kaggle Competition Link](Kaggle competition link](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/)

# COMMAND ----------

# MAGIC %md ##Install Library

# COMMAND ----------

pip install missingno

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# COMMAND ----------

# MAGIC %md ##Run Setup

# COMMAND ----------

# MAGIC %run ./../../global-setup $project_name=parkinsons-freezing_gait_prediction $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Show Tables
# MAGIC %sql
# MAGIC
# MAGIC SHOW TABLES LIKE 'park*'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Results

# COMMAND ----------

from ydata_profiling import ProfileReport

table = sql("SELECT * FROM parkinsons_defog_metadata").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
      "spearman": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file('parkinsons_defog_metadata.html')

# COMMAND ----------

table = sql("SELECT * FROM parkinsons_train_defog").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
      "spearman": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file('parkinsons_train_defog.html')

# COMMAND ----------

table = sql("SELECT * FROM parkinsons_train_notype").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file('parkinsons_train_notype.html')

# COMMAND ----------

table = sql("SELECT * FROM parkinsons_train_tdcsfog").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file("parkinsons_train_tdcsfog.html")

# COMMAND ----------

table = sql("SELECT * FROM parkinsons_defog_daily_metadata").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
      "spearman": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file("parkinsons_defog_daily_metadata.html")

# COMMAND ----------

table = sql("SELECT * FROM parkinsons_defog_tasks").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
      "spearman": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file("parkinsons_defog_tasks.html")

# COMMAND ----------

table = sql("SELECT * FROM parkinsons_fog_events").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
      "spearman": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file("parkinsons_fog_events.html")

# COMMAND ----------

table = sql("SELECT * FROM parkinsons_subjects").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
      "spearman": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file("parkinsons_subjects.html")

# COMMAND ----------

table = sql("SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_unlabeled").toPandas()

df_profile = ProfileReport(table,
  correlations={
      "auto": {"calculate": True},
      "pearson": {"calculate": True},
      "spearman": {"calculate": True},
  }, title="Profiling Report", progress_bar=False, infer_dtypes=False)

df_profile.to_file("parkinsons_unlabeled.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing data
# MAGIC
# MAGIC Recall from our data profile on the subjects table, UPDRS_Off is 24% null. Lets use the methods [here](https://www.kaggle.com/code/hafsaezzahraouy/dealing-with-missing-data-with-pyspark)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE grouped_subjects AS SELECT * FROM (
# MAGIC (SELECT 
# MAGIC   *,
# MAGIC   'tdcsfog' as FOG_group
# MAGIC FROM parkinsons_subjects s
# MAGIC WHERE Subject IN (SELECT DISTINCT Subject FROM parkinsons_tdcsfog_metadata))
# MAGIC UNION
# MAGIC (SELECT 
# MAGIC   *,
# MAGIC   'defog' as FOG_group
# MAGIC FROM parkinsons_subjects s
# MAGIC WHERE Subject IN (SELECT DISTINCT Subject FROM parkinsons_defog_metadata))
# MAGIC )

# COMMAND ----------

subjects_df = spark.table("grouped_subjects").toPandas()
subjects_df.info()

# COMMAND ----------

subjects_df.FOG_group = pd.factorize(subjects_df.FOG_group)[0]
subjects_df.Sex = pd.factorize(subjects_df.Sex)[0]

# COMMAND ----------

msno.bar(subjects_df)

# COMMAND ----------

msno.matrix(subjects_df)

# COMMAND ----------

num_df = pd.DataFrame(subjects_df.drop(columns=['Visit','Subject','Sex'])).copy(deep=True)
imputer = IterativeImputer()
num_df.iloc[:, :] = imputer.fit_transform(num_df)

# COMMAND ----------

subjects_df.UPDRSIII_Off = num_df.UPDRSIII_Off

# COMMAND ----------

msno.matrix(subjects_df)

# COMMAND ----------

spark.createDataFrame(subjects_df).write.mode("overwrite").saveAsTable("silver_subjects")

# COMMAND ----------


