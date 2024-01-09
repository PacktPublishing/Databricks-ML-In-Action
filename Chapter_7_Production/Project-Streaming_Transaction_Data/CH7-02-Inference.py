# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Production ML
# MAGIC
# MAGIC ## Synthetic data - Inference

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC %pip install --upgrade scikit-learn==1.4.0rc1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# DBTITLE 1,Batch inference from table
from databricks.feature_engineering import FeatureEngineeringClient
from mlia_utils.mlflow_funcs import get_latest_model_version
from pyspark.sql.types import *
import json
import pandas as pd
import mlflow
mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

model_name = f"{catalog}.{database_name}.packaged_transaction_model"

scoring_df = spark.table("prod_raw_transactions").drop("Label","_rescued_data").limit(100)

print(scoring_df)

print(f"Scoring model={model_name} version={get_latest_model_version(model_name)}")

scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{get_latest_model_version(model_name)}",
  df=scoring_df
)

display(scored)

# COMMAND ----------

# DBTITLE 1,Batch inference from new json data
from databricks.feature_engineering import FeatureEngineeringClient
from mlia_utils.mlflow_funcs import get_latest_model_version
from pyspark.sql.types import *
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
data = '[{"CustomerID":1240,"TransactionTimestamp": "2023-12-12T23:42:07.571Z","Product":"Product A","Amount":10.0}]'
scoring_df = pd.json_normalize(json.loads(data))
scoring_df["TransactionTimestamp"] = pd.to_datetime(scoring_df["TransactionTimestamp"])

print(f"Scoring model={model_name} version={get_latest_model_version(model_name)}")

scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{get_latest_model_version(model_name)}",
  df=spark.createDataFrame(scoring_df,schema=schema)
)

display(scored)

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

#tf = table format
def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/TransactionModel/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 
'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()
