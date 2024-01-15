# Databricks notebook source
import dlt

# COMMAND ----------

def build_autoloader_stream():
  raw_data_location = spark.conf.get('raw_data_location')
  return spark.readStream.format('cloudFiles') \
      .option("cloudFiles.format", "json") \
      .option("cloudFiles.inferColumnTypes","true") \
      .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
      .option("cloudFiles.schemaHints","CustomerID bigint, Amount double, TransactionTimestamp timestamp") \
      .load(f"{raw_data_location}")

# COMMAND ----------

def generate_table():
  table_name = spark.conf.get('table_name')
  @dlt.table(name=f'{table_name}',table_properties={"quality":"bronze"})
  def create_table(): 
    return build_autoloader_stream()

# COMMAND ----------

generate_table()
