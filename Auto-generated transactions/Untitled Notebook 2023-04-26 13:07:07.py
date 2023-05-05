# Databricks notebook source
./setup

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a Delta Live table
# MAGIC CREATE TABLE events
# MAGIC USING delta
# MAGIC AS SELECT *
# MAGIC FROM csv.`/path/to/csv/file`
# MAGIC
# MAGIC -- Create a stream to read from the Delta Live table
# MAGIC CREATE STREAM events_stream
# MAGIC USING delta
# MAGIC OPTIONS (
# MAGIC   'checkpointLocation' '/path/to/checkpoint/dir',
# MAGIC   'ignoreDeletes' 'true'
# MAGIC )
# MAGIC SELECT *
# MAGIC FROM events
# MAGIC
# MAGIC -- Define a query to check data quality
# MAGIC CREATE OR REPLACE TEMPORARY VIEW events_quality_check
# MAGIC AS SELECT 
# MAGIC   COUNT(*) AS num_records, 
# MAGIC   COUNT(DISTINCT event_id) AS num_unique_events, 
# MAGIC   MIN(event_time) AS min_event_time, 
# MAGIC   MAX(event_time) AS max_event_time
# MAGIC FROM events
# MAGIC
# MAGIC -- Write a query to continuously check data quality and write the results to a Delta Lake table
# MAGIC INSERT INTO delta.`/path/to/data_quality_table`
# MAGIC SELECT current_timestamp() AS check_time, * 
# MAGIC FROM events_quality_check
# MAGIC
# MAGIC -- Start the stream
# MAGIC START events_stream
