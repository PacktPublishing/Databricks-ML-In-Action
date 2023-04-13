# Databricks notebook source
# MAGIC %run ./setup

# COMMAND ----------

stemFilePath=cloud_storage_path
rawDataPath = "{}/autoloaderinput/".format(stemFilePath)

# COMMAND ----------

from pyspark.sql.functions import col, lit, max
from delta.tables import DeltaTable
import datetime

# COMMAND ----------

spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)

# COMMAND ----------

stream = spark.readStream \
  .format("cloudFiles") \
  .option("cloudFiles.format", "json") \
  .option("header", "true") \
  .option("cloudFiles.includeExistingFiles", "false")
  .option("cloudFiles.validateOptions", "true")
  .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
  .option("cloudFiles.schemaLocation", schema_location) \
  .load(raw_data_location) \
  .select(col("*")
          , col("_metadata")
  )

# COMMAND ----------

stream.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location) \
  .option("mergeSchema", "true") \
  .trigger(processingTime='5 seconds') \
  .start(target_delta_table_location)
