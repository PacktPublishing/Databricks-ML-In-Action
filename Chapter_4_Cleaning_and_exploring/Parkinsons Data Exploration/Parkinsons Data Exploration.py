# Databricks notebook source
# MAGIC %run ./../../global-setup $project_name=parkinsons-freezing_gait_prediction $catalog=lakehouse_in_action

# COMMAND ----------

pip install missingno fancyimpute

# COMMAND ----------

dbutils.library.restartPython()

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
# MAGIC ## Data File and Field Descriptions from Kaggle
# MAGIC
# MAGIC train/ Folder containing the data series in the training set within three subfolders: tdcsfog/, defog/, and notype/. Series in the notype folder are from the defog dataset but lack event-type annotations. The fields present in these series vary by folder.
# MAGIC * **Time** An integer timestep. Series from the tdcsfog dataset are recorded at 128Hz (128 timesteps per second), while series from the defog and daily series are recorded at 100Hz (100 timesteps per second).
# MAGIC * **AccV, AccML, and AccAP** Acceleration from a lower-back sensor on three axes: V - vertical, ML - mediolateral, AP - anteroposterior. Data is in units of m/s^2 for tdcsfog/ and g for defog/ and notype/.
# MAGIC * **StartHesitation**, Turn, Walking Indicator variables for the occurrence of each of the event types.
# MAGIC * **Event** Indicator variable for the occurrence of any FOG-type event. Present only in the notype series, which lack type-level annotations.
# MAGIC * **Valid** There were cases during the video annotation that were hard for the annotator to decide if there was an Akinetic (i.e., essentially no movement) FoG or the subject stopped voluntarily. Only event annotations where the series is marked true should be considered as unambiguous.
# MAGIC * **Task** Series were only annotated where this value is true. Portions marked false should be considered unannotated.
# MAGIC
# MAGIC Note that the Valid and Task fields are only present in the defog dataset. They are not relevant for the tdcsfog data.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Try out the Databricks Assistant.
# MAGIC Lets check that Subject & Visit make up the primary key for the Subject table
# MAGIC
# MAGIC Let's ask "How many pairs of Subject, Visit are in parkinsons_subjects"

# COMMAND ----------

# DBTITLE 1,This is what I asked for, but not what I wanted.
# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT Subject, Visit) AS pairs_count
# MAGIC FROM parkinsons_subjects

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try again with better wording "What are the Subject, Visit combinations and how many are there of each?"

# COMMAND ----------

# DBTITLE 1,This is the answer we were looking for.
# MAGIC %sql
# MAGIC SELECT Subject, Visit, count(*) as count FROM parkinsons_subjects GROUP BY Subject, Visit

# COMMAND ----------

# MAGIC %md
# MAGIC Asking the same question for the `tdcsfog_metadata` table shows Subject, Visit are not a primary key. In the documentation, it explains that Visit in the `defog_metadata` and the `tdcsfog_metadata` are different column meanings with the same name. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   Subject,
# MAGIC   Visit,
# MAGIC   COUNT(*) AS combo_count
# MAGIC FROM
# MAGIC   parkinsons_tdcsfog_metadata
# MAGIC GROUP BY
# MAGIC   Subject,
# MAGIC   Visit
# MAGIC ORDER BY
# MAGIC   Subject

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

import pandas as pd
subjects_df = table("grouped_subjects").toPandas()
subjects_df.info()

# COMMAND ----------

subjects_df.FOG_group = pd.factorize(subjects_df.FOG_group)[0]
subjects_df.Sex = pd.factorize(subjects_df.Sex)[0]

# COMMAND ----------

import missingno as msno
msno.bar(subjects_df)

# COMMAND ----------

msno.matrix(subjects_df)

# COMMAND ----------

from fancyimpute import IterativeImputer
mice_imputed = pd.DataFrame(subjects_df.UPDRSIII_Off).copy(deep=True)
mice_imputer = IterativeImputer()
mice_imputed.iloc[:, :] = mice_imputer.fit_transform(mice_imputed)

# COMMAND ----------

subjects_df.UPDRSIII_Off = mice_imputed.UPDRSIII_Off

# COMMAND ----------

msno.matrix(subjects_df)

# COMMAND ----------

spark.createDataFrame(subjects_df).write.mode("overwrite").saveAsTable("silver_subjects")

# COMMAND ----------


