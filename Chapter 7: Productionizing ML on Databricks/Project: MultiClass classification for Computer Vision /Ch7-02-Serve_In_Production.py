# Databricks notebook source
# MAGIC %pip install pytorch-lightning==2.1.2 evalidate==2.0.2 pillow==10.1.0 databricks-sdk==0.12.0

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

# MAGIC %run ./endpoint_sdk 

# COMMAND ----------

# MAGIC %md ## Get your wrapped model back

# COMMAND ----------

from mlflow import MlflowClient
import mlflow

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
model_name = f'{catalog}.{database_name}.cvops_model_mlaction'

client.set_registered_model_alias(model_name, "Champion", 1)

model_version_uri = f"models:/{model_name}@Champion"
# Or another option: model_version_uri = f"models:/{model_name}/1"
loaded_model_uc = mlflow.pyfunc.load_model(model_version_uri) # runiid / model_registry_name+version
# champion_version = client.get_model_version_by_alias("prod.ml_team.iris_model", "Champion")
latest_model = client.get_model_version_by_alias(model_name, "Champion")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create your serving endpoint 
# MAGIC
# MAGIC **Note** Here we have used UI to set the serving endpoint. You could also use REST API or SDK. 

# COMMAND ----------


import urllib
import json
import os 
import requests
import time 

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
serving_endpoint_name = f"cvops_model_mlaction_endpoint"

# NOTE HERE 
# We have wrapped REST API for the moment of this book creation 
# as SDK for Model Serving was not yet covering all functionalities. 
# Check under global-init. 
serving_client = EndpointApiClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": database_name,
    "table_name_prefix": serving_endpoint_name
    }

environment_vars={"DATABRICKS_TOKEN": "{{secrets/mlinaction/dl_token}}"}
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, model_name=model_name, model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True, auto_capture_config=auto_capture_config, environment_vars=environment_vars)


# COMMAND ----------

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')


# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json


def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = f"{db_host}/serving-endpoints/{serving_endpoint_name}/invocations"
  headers = {'Authorization': f'Bearer {db_token}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


# COMMAND ----------

import base64
import pandas as pd

images = spark.read.format("delta").load(val_delta_path).take(25)

b64image1 = base64.b64encode(images[0]["content"]).decode("ascii")
b64image2 = base64.b64encode(images[1]["content"]).decode("ascii")
b64image3 = base64.b64encode(images[2]["content"]).decode("ascii")
b64image4 = base64.b64encode(images[3]["content"]).decode("ascii")
b64image24 = base64.b64encode(images[24]["content"]).decode("ascii")

df_input = pd.DataFrame(
    [b64image1, b64image2, b64image3, b64image4, b64image24], columns=["data"])


# COMMAND ----------

score_model(df_input)

# COMMAND ----------


