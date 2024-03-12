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

display(model_version_details)

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference from table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

model_creation_time = model_version_details.creation_timestamp

scoring_df = sql(f"SELECT * FROM {table_name} WHERE TransactionTimestamp > {model_creation_time}").drop("Label")

print(f"Scoring model={model_name} version={model_version}")

scored = fe.score_batch(
  model_uri=f"models:/{model_name}/{model_version}",
  df=scoring_df,
  env_manager="conda"
)

display(scored)
