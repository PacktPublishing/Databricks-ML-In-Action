# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ##Synthetic data - Streaming as Delta into a table using Auto Loader + schema evolution
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ##Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_data_prod $catalog=hive_metastore

# COMMAND ----------

# DBTITLE 1,Variables
table_name = "synthetic_transactions"
raw_data_location = f"{cloud_storage_path}/data/"
schema_location = f"dbfs:/{cloud_storage_path}/{table_name}/schema"
checkpoint_location = f"dbfs:/{cloud_storage_path}/{table_name}/checkpoint"
database_name="lakehouse_prod_in_action"
sql("USE DATABASE lakehouse_prod_in_action")

# COMMAND ----------

# DBTITLE 1,Optimization settings and reduce the number of files that must be read to determine schema
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)
spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles",5)

# COMMAND ----------

# DBTITLE 1,Readstream
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
  .trigger(processingTime="10 seconds") \
  .toTable(tableName=table_name)

# COMMAND ----------

# DBTITLE 1,Viewing data in table while stream is running
display(sql(f"SELECT * FROM {table_name} ORDER BY TransactionTimestamp DESC LIMIT 10"))
