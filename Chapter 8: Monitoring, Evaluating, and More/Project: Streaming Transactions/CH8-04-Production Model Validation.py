# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 8: Monitoring, Evaluating, and More
# MAGIC
# MAGIC ## Model Validation
# MAGIC Use MLflow model validation API to test the model when registering a model

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

dbutils.widgets.text(name="model_name",defaultValue=f"{catalog}.{database_name}.packaged_transaction_model")
model_name = dbutils.widgets.get(name="model_name")

# COMMAND ----------

from mlia_utils.mlflow_funcs import get_latest_model_version
from mlflow.tracking import MlflowClient

import mlflow

# Set UC As Default Registry #
mlflow.set_registry_uri("databricks-uc")
mlfclient = MlflowClient()

validation_results = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model details

# COMMAND ----------

model_details = mlfclient.get_registered_model(model_name)
model_version = str(get_latest_model_version(model_name))
model_version_details = mlfclient.get_model_version(name=model_name, version=model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Check for tags

# COMMAND ----------

assert 'validation_status' in model_version_details.tags.keys(), f"the model, model={model_name}, specfied does not have validation_status tag"
assert model_version_details.tags['validation_status'] == 'needs_tested', f"the latest version, version={model_version}, of model, model={model_name} is not tagged as validation_status=needs_tested"

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model validation
# MAGIC Create tests that can be applied to all models
# MAGIC
# MAGIC ####Documentation

# COMMAND ----------

if model_details.description == "":
  print("No model description found. Please add a model description to pass this test.")
  validation_results['description'] = 0
elif len(model_details.description) < 15:
  print("Model description is too short. Perhaps include a note about the business impact.")
  validation_results['description'] = 0
else:
  validation_results['description'] = 1

# COMMAND ----------

if model_details.tags == {}:
  print("No model tags found. It is team deployment policy that all models in production have at least the project name tag.")
  validation_results['project_tag'] = 0
elif not 'project' in model_details.tags.keys():
  print("No project tag found. It is team deployment policy that all models in production have the project name tag.")
  validation_results['project_tag'] = 0
elif not model_details.tags['project'] in possible_projects:
  print(f"Your project name was not recognized. Perhaps it was misspelled. Project name, project='{model_details.tags['project']}, is not in {possible_projects}")
  validation_results['project_tag'] = 0
else:
  validation_results['project_tag'] = 1

# COMMAND ----------

if sum(validation_results.values()) == len(validation_results.values()):
  mlfclient.set_model_version_tag(name=model_name, key="validation_status", value="passed_tests", version=model_version)
  print(f"Success! Your model, {model_name} version {model_version}, passed all tests.")
else:
  mlfclient.set_model_version_tag(name=model_name, key="validation_status", value="failed_tests", version=model_version)
  print(f"Fail! Check the results to determine which test(s) the model failed. {validation_results}")
