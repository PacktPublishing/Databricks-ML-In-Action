# Databricks notebook source
# MAGIC %md
# MAGIC ## Synthetic Data

# COMMAND ----------

dbutils.widgets.dropdown(name='Reset', defaultValue='False', choices=['True', 'False'], label="Reset Checkpoint and Schema")

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_data $catalog=lakehouse_in_action

# COMMAND ----------

table_name = "synthetic_transactions"
raw_data_location = f"{spark_storage_path}/data"
schema_location = f"{spark_storage_path}/{table_name}/schema"
checkpoint_location = f"{spark_storage_path}/{table_name}/checkpoint"

# COMMAND ----------

# DBTITLE 1,Use to reset for fresh table, schema, checkpoints
if bool(dbutils.widgets.get('Reset')):
  dbutils.fs.rm(schema_location, True)
  dbutils.fs.rm(checkpoint_location, True)
  sql(f"DROP TABLE IF EXISTS {table_name}")

# COMMAND ----------

# DBTITLE 1,Optimization settings and reduce the number of files that must be read to determine schema
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)
spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles",5)

# COMMAND ----------

# DBTITLE 1,Readstream and writestream
stream = spark.readStream \
  .format("cloudFiles") \
  .option("cloudFiles.format", "json") \
  .option("cloudFiles.schemaHints","CustomerID bigint, Amount double, TransactionTimestamp timestamp") \
  .option("cloudFiles.inferColumnTypes","true") \
  .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
  .option("cloudFiles.schemaLocation", schema_location) \
  .load(raw_data_location) \
  .select("*") \
  .writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location) \
  .option("mergeSchema", "true") \
  .trigger(processingTime='10 seconds') \
  .toTable(tableName=table_name)    
  .trigger(availableNow=True) 

# COMMAND ----------

# DBTITLE 1,Viewing data in table while stream is running
display(sql(f"SELECT * FROM {table_name} ORDER BY TransactionTimestamp DESC LIMIT 10"))

# COMMAND ----------


