# Databricks notebook source
# MAGIC %run ./../setup

# COMMAND ----------

# DBTITLE 1,Show Tables
# MAGIC %sql
# MAGIC
# MAGIC SHOW TABLES LIKE 'park*'

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT isnull(ddm.Id) as daily_defog_metadata, isnull(dm.Id) as defog_metadata, isnull(dt.Id) as defog_tasks, isnull(fe.Id) as fog_events, isnull(tm.Id) as tdcsfog_metadata, isnull(u.id) as unlabeled, isnull(td.Id) as train_defog, isnull(tnt.Id) as train_notype, count(1) FROM parkinsons_defog_daily_metadata ddm
# MAGIC FULL OUTER JOIN parkinsons_defog_metadata dm ON ddm.Id = dm.Id
# MAGIC FULL OUTER JOIN parkinsons_fog_events fe ON (fe.Id = dt.Id OR fe.Id = dm.Id OR fe.Id = ddm.Id)
# MAGIC FULL OUTER JOIN parkinsons_tdcsfog_metadata tm ON (tm.Id = fe.Id OR tm.Id = dm.Id OR tm.Id = ddm.Id)
# MAGIC FULL OUTER JOIN parkinsons_unlabeled u ON (u.id = tm.Id OR u.Id = fe.Id OR u.Id = dm.Id OR u.Id = ddm.Id)
# MAGIC FULL OUTER JOIN parkinsons_train_defog td ON (td.id = tm.Id OR td.Id = fe.Id OR td.Id = dm.Id OR td.Id = ddm.Id)
# MAGIC FULL OUTER JOIN parkinsons_train_notype tnt ON (tnt.id = tm.Id OR tnt.Id = fe.Id OR tnt.Id = dm.Id OR tnt.Id = ddm.Id)
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

# MAGIC %md
# MAGIC ## Tiny data

# COMMAND ----------

sub_id = "c432df"
sql(f"""CREATE OR REPLACE TABLE parkinsons_tiny_subjects AS (SELECT * FROM parkinsons_subjects WHERE Subject = '{sub_id}')""")
sql(f"""CREATE OR REPLACE TABLE parkinsons_tiny_defog_metadata AS (SELECT * FROM parkinsons_defog_metadata WHERE Subject = '{sub_id}')""")
sql(f"""CREATE OR REPLACE TABLE parkinsons_tiny_tdcsfog_metadata AS (SELECT * FROM parkinsons_tdcsfog_metadata WHERE Subject = '{sub_id}')""")
sql(f"""CREATE OR REPLACE TABLE parkinsons_tiny_defog_daily_metadata AS (SELECT * FROM parkinsons_defog_daily_metadata WHERE Subject = '{sub_id}')""")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_defog_daily_metadata

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_defog_metadata

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_tdcsfog_metadata
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_defog_metadata ev 
# MAGIC left JOIN parkinsons_defog_tasks tdm ON ev.id = tdm.Id

# COMMAND ----------

dfa = sql("SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_defog_metadata")
dfb = sql("SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_defog_daily_metadata")
dfc = sql("SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_subjects")

df1 = dfa.unionByName(dfb, allowMissingColumns=True)
df1 = df1.join(dfc,['Subject','Visit'],'full')
display(df1)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT *, row_number() OVER (PARTITION BY `Id` ORDER BY Init) TimeStep FROM parkinsons_fog_events WHERE `Id` IN ('48081794eb','ad8e83242a','60dfb26b2c')

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT * FROM parkinsons_fog_events WHERE `Id` IN ('48081794eb','ad8e83242a','60dfb26b2c')

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT * FROM parkinsons_defog_tasks WHERE `Id` IN ('48081794eb','ad8e83242a','60dfb26b2c')

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;
# MAGIC
# MAGIC SELECT * FROM parkinsons_unlabeled WHERE `Id` IN ('48081794eb','ad8e83242a','60dfb26b2c')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   events.`Id`,
# MAGIC   events.Init,
# MAGIC   events.Completion,
# MAGIC   events.`Type`,
# MAGIC   events.Kinetic,
# MAGIC   df.`Time`,
# MAGIC   df.Walking,
# MAGIC   df.Turn,
# MAGIC   df.StartHesitation,
# MAGIC   df.Valid,
# MAGIC   df.Task,
# MAGIC   df.AccV,
# MAGIC   df.AccML,
# MAGIC   df.AccAP
# MAGIC FROM
# MAGIC   hive_metastore.lakehouse_in_action.parkinsons_fog_events events
# MAGIC JOIN hive_metastore.lakehouse_in_action.parkinsons_train_defog df ON events.Id = df.id
# MAGIC WHERE df.id = 'da05ad7058' --IN ('48081794eb','ad8e83242a','60dfb26b2c')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   events.`Type`,
# MAGIC   events.Kinetic,
# MAGIC   df.Walking,
# MAGIC   df.Turn,
# MAGIC   df.StartHesitation,
# MAGIC   df.Valid,
# MAGIC   df.Task,
# MAGIC   count(*) as count
# MAGIC FROM
# MAGIC   hive_metastore.lakehouse_in_action.parkinsons_fog_events events
# MAGIC JOIN hive_metastore.lakehouse_in_action.parkinsons_train_defog df ON events.Id = df.id
# MAGIC WHERE df.
# MAGIC GROUP BY ALL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC  *
# MAGIC FROM
# MAGIC   parkinsons_tdcsfog_metadata m
# MAGIC   JOIN parkinsons_subjects sub ON m.Subject = sub.Subject AND m.Visit = sub.Visit

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC *, 
# MAGIC CASE 
# MAGIC WHEN (Task == true AND Valid == true AND (StartHesitation + Turn + Walking) > 0)
# MAGIC THEN 1 ELSE 0 END AS EventInProgress
# MAGIC FROM lakehouse_in_action.parkinsons_train_defog

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 

df = sql('select *,  from lakehouse_in_action.parkinsons_train_defog where Valid == TRUE AND Task == TRUE AND (StartHesitation + Turn + Walking) > 0')

df_index = df.select("*").withColumn("id", monotonically_increasing_id())
display(df_index)
#df_index.write.mode("overwrite").saveAsTable("unambiguous_indexed")

# COMMAND ----------

# MAGIC %md We could use bamboolib here, or we could use the pandas profiler

# COMMAND ----------


