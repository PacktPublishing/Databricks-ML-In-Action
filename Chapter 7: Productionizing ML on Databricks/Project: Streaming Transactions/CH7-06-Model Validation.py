# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Productionizing ML on Databricks
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

from mlflow.client import MlflowClient
import mlflow

# Set UC As Default Registry #
mlflow.set_registry_uri("databricks-uc")
mlfclient = mlflow.tracking.MlflowClient()

validation_results = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model details

# COMMAND ----------

model_details = mlfclient.get_registered_model(model_name)
model_version_info = mlfclient.search_model_versions(f"name='{model_name}'")[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ####Check for aliases

# COMMAND ----------

assert 'needs_tested' in model_details.aliases.keys(), f"the model, model={model_name}, specfied does not have any version with alias 'needs_tested'"
assert model_details.aliases['needs_tested'] == model_version_info.version, f"the latest version, version={model_version_info.version}, of model, model={model_name} is not aliased as 'needs_testing'"

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
  mlfclient.set_registered_model_alias(name=model_name, alias="passed_tests", version=model_version_info.version)
  mlfclient.delete_registered_model_alias(name=model_name, alias="needs_tested")
  if 'failed_tests' in model_details.aliases.keys():
    mlfclient.delete_registered_model_alias(name=model_name, alias="failed_tests")
  print(f"Success! Your model, {model_name} version {model_version_info.version}, passed all tests.")
else:
  if 'passed_tests' in model_details.aliases.keys():
    mlfclient.delete_registered_model_alias(name=model_name, alias="passed_tests")
  mlfclient.set_registered_model_alias(name=model_name, alias="failed_tests", version=model_version_info.version)
  print(f"Fail! Check the results to determine which test(s) the model failed. {validation_results}")
