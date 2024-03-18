# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 8: Monitoring, Evaluating, and More
# MAGIC
# MAGIC ## Labeled data - Streaming as Delta into a table using Auto Loader

# COMMAND ----------

# MAGIC %md ##Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

# DBTITLE 1,Variables
dbutils.widgets.text('raw_table_name','transaction_labels','Enter table name for the raw delta')
table_name = dbutils.widgets.get('raw_table_name')
raw_data_location = f"{volume_file_path}/{table_name}/data/" 
schema_location = f"{volume_file_path}/{table_name}/schema"
checkpoint_location = f"{volume_file_path}/{table_name}/checkpoint"

inference_table = f"{catalog}.{database_name}.packaged_transaction_model_predictions"

# COMMAND ----------

# DBTITLE 1,Use to reset for fresh table, schema, checkpoints
if not spark.catalog.tableExists(table_name):
  sql(f"""CREATE TABLE {table_name} TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

# COMMAND ----------

# DBTITLE 1,Optimization settings and reduce the number of files that must be read to determine schema
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)
spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles",1)
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", True)

# COMMAND ----------

# DBTITLE 1,Readstream
stream = spark.readStream \
  .format("cloudFiles") \
  .option("cloudFiles.format", "json") \
  .option("cloudFiles.schemaHints","CustomerID int, TransactionTimestamp timestamp, Label int") \
  .option("cloudFiles.inferColumnTypes","true") \
  .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
  .option("cloudFiles.schemaLocation", schema_location) \
  .load(raw_data_location) \
  .select("*")
  
(stream.writeStream
  .format("delta")
  .outputMode("append")
  .option("checkpointLocation", checkpoint_location)
  .option("mergeSchema", "true")
  .trigger(availableNow=True)
  .toTable(tableName=table_name))

# COMMAND ----------

#for each batch / incremental update from readStream, 
# we'll run a MERGE on the inference table
def merge_stream(df, i):
  df.createOrReplaceTempView("labels_cdc_microbatch")
  #We run the merge (upsert or delete)
  df._jdf.sparkSession()\
    .sql(f"""MERGE INTO {inference_table} target
        USING
        (SELECT CustomerID, TransactionTimestamp, Label as actual_label 
        FROM labels_cdc_microbatch) as source
        ON source.CustomerID = target.CustomerID 
        AND source.TransactionTimestamp = target.TransactionTimestamp
        WHEN MATCHED THEN UPDATE SET target.actual_label == source.actual_label
        """)
  
if spark.catalog.tableExists(inference_table):
  (stream.writeStream
    .foreachBatch(merge_stream)
    .option("checkpointLocation", volume_file_path+inference_table+"/checkpoint_cdc")
    .trigger(availableNow=True)
    .start()
  )
