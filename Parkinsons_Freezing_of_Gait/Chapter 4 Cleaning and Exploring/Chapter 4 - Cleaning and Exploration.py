# Databricks notebook source
# MAGIC %run ./../setup

# COMMAND ----------

# DBTITLE 1,Show Tables
# MAGIC %sql
# MAGIC
# MAGIC SHOW TABLES LIKE 'park*'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joining datasets

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_defog_daily_metadata ddm
# MAGIC INNER JOIN parkinsons_unlabeled other ON ddm.Id = other.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_train_tdcsfog ddm
# MAGIC INNER JOIN parkinsons_tdcsfog_metadata other ON ddm.Id = other.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_test_tdcsfog ddm
# MAGIC INNER JOIN parkinsons_tdcsfog_metadata other ON ddm.Id = other.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_train_defog a
# MAGIC INNER JOIN parkinsons_defog_metadata b ON a.Id = b.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_fog_events ddm
# MAGIC INNER JOIN parkinsons_train_notype other ON ddm.Id = other.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_defog_metadata ddm
# MAGIC INNER JOIN parkinsons_train_notype other ON ddm.Id = other.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_fog_events a
# MAGIC INNER JOIN parkinsons_train_defog b ON a.Id = b.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_fog_events a
# MAGIC INNER JOIN parkinsons_train_tdcsfog b ON a.Id = b.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC
# MAGIC SELECT count(1) FROM parkinsons_defog_tasks a
# MAGIC INNER JOIN parkinsons_train_notype b ON a.Id = b.Id

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC
# MAGIC SELECT isnull(a.Id) as tasks, isnull(b.Id) as defog_metadata, count(1) FROM parkinsons_defog_tasks a
# MAGIC FULL JOIN parkinsons_defog_metadata b ON a.Id = b.Id
# MAGIC GROUP BY ALL

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC SELECT
# MAGIC   isnull(ddm.Id) as daily_defog_metadata,
# MAGIC   isnull(dm.Id) as defog_metadata,
# MAGIC   isnull(dt.Id) as defog_tasks,
# MAGIC   isnull(fe.Id) as fog_events,
# MAGIC   count(1)
# MAGIC FROM
# MAGIC   parkinsons_defog_daily_metadata ddm FULL
# MAGIC   OUTER JOIN parkinsons_defog_metadata dm ON ddm.Id = dm.Id FULL
# MAGIC   OUTER JOIN parkinsons_defog_tasks dt ON (
# MAGIC     dt.Id = ddm.Id
# MAGIC     or dt.Id = dm.Id
# MAGIC   ) FULL
# MAGIC   OUTER JOIN parkinsons_fog_events fe ON (
# MAGIC     fe.Id = dt.Id
# MAGIC     OR fe.Id = dm.Id
# MAGIC     OR fe.Id = ddm.Id
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   ALL

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
# MAGIC The Subject & Visit make up the primary key for the Subject table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Subject, Visit, count(*) as count FROM hive_metastore.lakehouse_in_action.parkinsons_subjects GROUP BY Subject, Visit

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT StartHesitation, Turn, Walking, Valid, Task, COUNT(*) as cnt FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog GROUP BY ALL ORDER BY cnt DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT int(`Time`) as Time, int(StartHesitation) as StartHesitation, int(Walking) as Walking, int(Turn) as Turn, `id` FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog

# COMMAND ----------


