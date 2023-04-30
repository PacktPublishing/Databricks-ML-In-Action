# Databricks notebook source
# MAGIC %run ./setup

# COMMAND ----------

raw_data_location = f"{spark_storage_path}/autoloader/"
schema_location = f"{raw_data_location}/schema"
checkpoint_location = f"{raw_data_location}/checkpoint"

table_name = "transactions"
target_delta_table_location = f"{spark_storage_path}/{table_name}/table_data"

# COMMAND ----------

# dbutils.fs.rm(schema_location, True)
# dbutils.fs.rm(checkpoint_location, True)
# sql(f"DROP TABLE IF EXISTS {table_name}")
# dbutils.fs.mkdirs(target_delta_table_location)

# COMMAND ----------

spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)

# COMMAND ----------

stream = spark.readStream \
  .format("cloudFiles") \
  .option("cloudFiles.format", "json") \
  .option("header", "true") \
  .option("cloudFiles.includeExistingFiles", "false") \
  .option("cloudFiles.schemaEvolutionMode", "rescue") \
  .option("cloudFiles.schemaLocation", schema_location) \
  .load(raw_data_location)

# COMMAND ----------

stream.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location) \
  .option("mergeSchema", "true") \
  .trigger(once=True) \
  .start(target_delta_table_location)

# COMMAND ----------

display(stream)

# COMMAND ----------


