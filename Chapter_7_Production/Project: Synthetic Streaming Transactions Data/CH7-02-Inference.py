# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Production ML
# MAGIC
# MAGIC ## Synthetic data - Inference

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

display(spark.read.format("text").load('/Volumes/ml_in_action/synthetic_transactions/files/raw_transactions/data'))

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from mlia_utils.mlflow_funcs import get_latest_model_version
import json
import pandas as pd
import mlflow
mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

model_name = f"{catalog}.{database_name}.packaged_transaction_model"
schema = StructType([
    StructField("CustomerID", IntegerType(), False),
    StructField("TransactionTimestamp", TimestampType(), False),
    StructField("Product", StringType(), True),
    StructField("Amount", FloatType(),False)
])
data = '[{"CustomerID":1240,"TransactionTimestamp":"2023-12-12T23:42:07.571Z","Product":"Product A","Amount":10.0}]'
print(json.loads(data))
scoring_df = pd.json_normalize(json.loads(data))


scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{get_latest_model_version(model_name)}",
  df=spark.createDataFrame(scoring_df,schema=schema)
)

display(scored)
