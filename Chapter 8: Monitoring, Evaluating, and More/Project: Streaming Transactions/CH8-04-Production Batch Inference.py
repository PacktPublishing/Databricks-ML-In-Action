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

inference_table = f"{catalog}.{database_name}.packaged_transaction_model_predictions"
outputPath = f"{volume_file_path}/{inference_table}/streaming_outputs/"

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

from pyspark.sql.types import StringType, StructField, StructType, IntegerType, FloatType, TimestampType

# The schema for the incoming records
schema = StructType([
    StructField("Source", StringType(), True),
    StructField("TransactionTimestamp", StringType(), True),
    StructField("CustomerID", IntegerType(), True),
    StructField("Amount", FloatType(), True),
    StructField("Product", StringType(), True),
    StructField("Label", IntegerType(), True)
])

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import *

fe = FeatureEngineeringClient()

inputDf = (spark.readStream
  .format("delta")
  .schema(schema)
  .table(table_name)
  .selectExpr("CustomerID","Product","Amount","cast(TransactionTimestamp as timestamp) TransactionTimestamp")
  .display())

def batchInference(batch_df, batch_id):
    # Convert the dataset to a dataframe for merging
    scoring_df = batch_df

    scored = fe.score_batch(
      model_uri=f"models:/{model_name}/{model_version}",
      df=scoring_df
    )

#     # Write the results to a Delta Lake table
#     scored.write.format("delta").mode("append").save(f"{outputPath}/scored_data")

(inputDf.writeStream
  .foreachBatch(batchInference)
  .option("checkpointLocation", f"{outputPath}/checkpoint") 
  .
  .queryName("batchInference")
  .outputMode("append")
  .start())

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql import *

inputDf = spark.readStream \
  .format("delta") \
  .schema(schema) \
  .table(table_name) \
  .select("*") #, "cast(TransactionTimestamp as timestamp) TransactionTimestamp")

def batchInference(newRows):
    # Convert the dataset to a dataframe for merging
    scoring_df = newRows.toDF()
    
    scored = fe.score_batch(
      model_uri=f"models:/{model_name}/{model_version}",
      df=scoring_df,
      env_manager="conda"
    )
    return scored


inputDf.writeStream \
  .foreachBatch(batchInference) \
  .option("checkpointLocation", f"{outputPath}/checkpoint") \
  .trigger(processingTime="180 seconds") \
  .queryName("batchInference") \
  .outputMode("append") \
  .toTable(inference_table)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

stream = spark.readStream \
  .format("delta") \
    .schema(schema) \
    .table(inputTable) \
    .select("CustomerID", "cast(TransactionTimestamp as timestamp) TransactionTimestamp") \
  
  
  
  
  .select("*") \
  .writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location) \
  .option("mergeSchema", "true") \
  .trigger(processingTime="10 seconds") \
  .toTable(tableName=table_name)


scoring_df = sql(f"SELECT * FROM {table_name} WHERE TransactionTimestamp > {model_creation_time}").drop("Label")

print(f"Scoring model={model_name} version={model_version}")

scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{model_version}",
  df=scoring_df,
  env_manager="conda"
)

display(scored)
