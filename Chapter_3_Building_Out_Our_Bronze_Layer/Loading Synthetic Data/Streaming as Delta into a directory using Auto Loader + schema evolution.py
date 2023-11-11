# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 3: Building out our Bronze Layer
# MAGIC
# MAGIC ## Synthetic Data - Streaming as Delta into a directory using Auto Loader + schema evolution

# COMMAND ----------

# DBTITLE 1,Create Checkpoint and Schema reset widget
dbutils.widgets.dropdown(name='Reset', defaultValue='False', choices=['True', 'False'], label="Reset Checkpoint and Schema")

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_data $catalog=hive_metastore

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# DBTITLE 1,Variables
#table_name = "synthetic_transactions"
raw_data_location = f"{cloud_storage_path}/data/"
destination_location = f"{cloud_storage_path}/delta/"
schema_location = f"dbfs:/{cloud_storage_path}/directory/schema"
checkpoint_location = f"dbfs:/{cloud_storage_path}/directory/checkpoint"

# COMMAND ----------

# DBTITLE 1,Use to reset for fresh table, schema, checkpoints
if bool(dbutils.widgets.get('Reset')):
  dbutils.fs.rm(schema_location, True)
  dbutils.fs.rm(checkpoint_location, True)
  dbutils.fs.rm(destination_location, True)
  #sql(f"DROP TABLE IF EXISTS {table_name}")

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
  .start(destination_location)

# COMMAND ----------

# DBTITLE 1,Viewing data in directory while stream is running
df = spark.read.format("delta").load(destination_location)
display(df.orderBy(col("TransactionTimestamp").desc()))
