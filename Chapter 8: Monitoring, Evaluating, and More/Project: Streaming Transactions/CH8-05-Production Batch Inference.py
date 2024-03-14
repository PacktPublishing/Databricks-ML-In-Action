# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 8: Monitoring, Evaluating, and More
# MAGIC
# MAGIC ## Batch Inference

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC %pip install --upgrade scikit-learn==1.4.0rc1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

dbutils.widgets.text(name="model_name",defaultValue=f"{catalog}.{database_name}.packaged_transaction_model")
model_name = dbutils.widgets.get(name="model_name")

dbutils.widgets.text('raw_table_name','prod_transactions','Enter raw table name')
table_name = dbutils.widgets.get('raw_table_name')

ft_name = "product_3minute_max_price_ft"
inference_table = f"{catalog}.{database_name}.packaged_transaction_model_predictions"

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model details

# COMMAND ----------

from mlia_utils.mlflow_funcs import get_latest_model_version
from mlflow.tracking import MlflowClient

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlfclient = MlflowClient()

model_details = mlfclient.get_registered_model(model_name)
model_version = str(get_latest_model_version(model_name))
model_version_details = mlfclient.get_model_version(name=model_name, version=model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference from table

# COMMAND ----------

min_time = sql(f"SELECT MIN(LookupTimestamp) FROM {ft_name}").collect()[0][0]
max_time = sql(f"SELECT MAX(LookupTimestamp) FROM {ft_name}").collect()[0][0]

if not spark.catalog.tableExists(inference_table) or spark.table(tableName=inference_table).isEmpty():
  scoring_df = sql(f"""
                   SELECT Amount,CustomerID,Product,TransactionTimestamp FROM {table_name} 
                   WHERE TransactionTimestamp <= '{max_time}' AND TransactionTimestamp >= '{min_time}'
                   """)
  sql(f"""CREATE TABLE IF NOT EXISTS {inference_table} (CustomerID INT NOT NULL, TransactionTimestamp TIMESTAMP NOT NULL, Label INT) 
  TBLPROPERTIES (delta.enableChangeDataFeed = true, delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true)""")
else:
  last_inf_time = sql(f"SELECT MAX(LookupTimestamp) FROM {inference_table}").collect()[0][0]  
  scoring_df = sql(f"""
                   SELECT Amount,CustomerID,Product,TransactionTimestamp FROM {table_name} 
                   WHERE TransactionTimestamp <= '{max_time}' AND TransactionTimestamp >= '{min_time}'
                   AND TransactionTimestamp > '{last_inf_time}'
                   """)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import *

fe = FeatureEngineeringClient()


scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{model_version}",
  df=scoring_df
)
scored.withColumn(colName="actual_label",col=lit(-1)).write.mode('append').format('delta').saveAsTable(inference_table)
