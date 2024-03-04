# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Productionizing ML on Databricks
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

model_name = f"{catalog}.{database_name}.packaged_transaction_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference from table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from mlia_utils.mlflow_funcs import get_latest_model_version
import mlflow
mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

scoring_df = sql("SELECT * FROM raw_transactions ORDER BY TransactionTimestamp DESC").drop("Label").limit(100)

print(f"Scoring model={model_name} version={get_latest_model_version(model_name)}")

scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{get_latest_model_version(model_name)}",
  df=scoring_df,
  env_manager="conda"
)

display(scored)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Batch inference from new json data

# COMMAND ----------

# DBTITLE 0,Batch inference from new json data
from pyspark.sql.types import *
import pandas as pd
import json

schema = StructType([
    StructField("CustomerID", IntegerType(), False),
    StructField("TransactionTimestamp", TimestampType(), False),
    StructField("Product", StringType(), True),
    StructField("Amount", FloatType(),False)
])
data = '[{"CustomerID":1240,"TransactionTimestamp": "2024-01-11T03:10:17.416+00:00","Product":"Product A","Amount":10.0}]'
scoring_df = pd.json_normalize(json.loads(data))
scoring_df["TransactionTimestamp"] = pd.to_datetime(scoring_df["TransactionTimestamp"])

print(f"Scoring model={model_name} version={get_latest_model_version(model_name)}")

scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{get_latest_model_version(model_name)}",
  df=spark.createDataFrame(scoring_df,schema=schema),
  env_manager="conda"
)

display(scored)
