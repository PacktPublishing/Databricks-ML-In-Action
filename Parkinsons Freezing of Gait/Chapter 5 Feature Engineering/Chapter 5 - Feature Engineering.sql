-- Databricks notebook source
-- MAGIC %run ./../setup

-- COMMAND ----------

USE catalog hive_metastore;
USE lakehouse_in_action;

CREATE OR REPLACE TABLE parkinsons_tiny_fe_n_metadata_w_time AS (SELECT *, row_number() OVER (PARTITION BY `Id` ORDER BY Init) TimeStep FROM parkinsons_tiny_fe_n_metadata)

-- COMMAND ----------

CREATE OR REPLACE TABLE hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids AS (SELECT * FROM (SELECT DISTINCT id FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog) TABLESAMPLE (5 ROWS));
SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col
-- MAGIC
-- MAGIC subs = sql("""SELECT Subject FROM hive_metastore.lakehouse_in_action.parkinsons_defog_metadata WHERE id IN (SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids)""")
-- MAGIC subs = subs.select(col("Subject")).rdd.flatMap(lambda x: x).collect()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC sql(
-- MAGIC     f"""CREATE OR REPLACE TABLE parkinsons_tiny_subjects AS (SELECT * FROM parkinsons_subjects WHERE Subject IN {tuple(subs)})"""
-- MAGIC )
-- MAGIC sql(
-- MAGIC     f"""CREATE OR REPLACE TABLE parkinsons_tiny_defog_metadata AS (SELECT * FROM parkinsons_defog_metadata WHERE Subject IN {tuple(subs)})"""
-- MAGIC )
-- MAGIC sql(
-- MAGIC     f"""CREATE OR REPLACE TABLE parkinsons_tiny_tdcsfog_metadata AS (SELECT * FROM parkinsons_tdcsfog_metadata WHERE Subject IN {tuple(subs)})"""
-- MAGIC )
-- MAGIC
-- MAGIC

-- COMMAND ----------

SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog tdf
WHERE tdf.id IN (SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_event_ids)

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC CREATE
-- MAGIC OR REPLACE TABLE parkinsons_tiny_fe_n_metadata AS (
-- MAGIC   SELECT
-- MAGIC     fe.`Id`,
-- MAGIC     fe.Init,
-- MAGIC     fe.Completion,
-- MAGIC     fe.Kinetic,
-- MAGIC     fe.`Type`,
-- MAGIC     dm.Subject,
-- MAGIC     dm.Visit,
-- MAGIC     null as Test,
-- MAGIC     dm.Medication
-- MAGIC   from
-- MAGIC     parkinsons_fog_events fe
-- MAGIC     JOIN parkinsons_tiny_defog_metadata dm ON fe.Id = dm.Id
-- MAGIC   UNION
-- MAGIC   SELECT
-- MAGIC     fe.`Id`,
-- MAGIC     fe.Init,
-- MAGIC     fe.Completion,
-- MAGIC     fe.Kinetic,
-- MAGIC     fe.`Type`,
-- MAGIC     tm.Subject,
-- MAGIC     tm.Visit,
-- MAGIC     tm.Test,
-- MAGIC     tm.Medication
-- MAGIC   from
-- MAGIC     parkinsons_fog_events fe
-- MAGIC     JOIN parkinsons_tiny_tdcsfog_metadata tm ON fe.Id = tm.Id
-- MAGIC )

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %sql
-- MAGIC CREATE
-- MAGIC OR REPLACE TABLE parkinsons_tiny_tasks AS (
-- MAGIC   SELECT
-- MAGIC     *
-- MAGIC   from
-- MAGIC     parkinsons_defog_tasks
-- MAGIC   WHERE
-- MAGIC     `Id` IN (SELECT * FROM parkinsons_tiny_event_ids)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC SELECT
-- MAGIC   events.`Type`,
-- MAGIC   events.Kinetic,
-- MAGIC   df.Walking,
-- MAGIC   df.Turn,
-- MAGIC   df.StartHesitation,
-- MAGIC   df.Valid,
-- MAGIC   df.Task,
-- MAGIC   count(*) as count
-- MAGIC FROM
-- MAGIC   hive_metastore.lakehouse_in_action.parkinsons_fog_events events
-- MAGIC JOIN hive_metastore.lakehouse_in_action.parkinsons_train_defog df ON events.Id = df.id
-- MAGIC GROUP BY ALL

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC SELECT 
-- MAGIC *, 
-- MAGIC CASE 
-- MAGIC WHEN (Task == true AND Valid == true AND (StartHesitation + Turn + Walking) > 0)
-- MAGIC THEN 1 ELSE 0 END AS EventInProgress
-- MAGIC FROM lakehouse_in_action.parkinsons_train_defog

-- COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 

df = sql('select *,  from lakehouse_in_action.parkinsons_train_defog where Valid == TRUE AND Task == TRUE AND (StartHesitation + Turn + Walking) > 0')

df_index = df.select("*").withColumn("id", monotonically_increasing_id())
display(df_index)
#df_index.write.mode("overwrite").saveAsTable("unambiguous_indexed")

-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------



-- COMMAND ----------


