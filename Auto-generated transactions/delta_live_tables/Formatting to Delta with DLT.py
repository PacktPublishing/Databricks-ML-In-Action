# Databricks notebook source
import dlt
from pyspark.sql import functions as F

# COMMAND ----------

def build_autoloader_stream():
  raw_data_location = spark.conf.get('raw_data_location')
  schema_location = spark.conf.get('schema_location')
  return spark.readStream.format('cloudFiles') \
      .option("cloudFiles.format", "json") \
      .option("cloudFiles.inferColumnTypes","true") \
      .option("cloudFiles.schemaEvolutionMode", "addNewColumns") \
      .option("cloudFiles.schemaLocation", f"{schema_location}") \
      .option("cloudFiles.schemaHints","CustomerID bigint, Amount double, TransactionTimestamp timestamp") \
      .load(f"{raw_data_location}")

# COMMAND ----------

def generate_table():
  table_name = spark.conf.get('table_name')
  @dlt.table(name=f'{table_name}')
  def create_table(): 
    return build_autoloader_stream()

# COMMAND ----------

generate_table()
