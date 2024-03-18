# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 databricks-sdk==0.12.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=rag_chatbot

# COMMAND ----------

import mlflow
from mlia_utils.mlflow_funcs import * 

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

model_name = f"{catalog}.{database_name}.mlaction_chatbot_model"
serving_endpoint_name = "mlaction_chatbot_model_hf"
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()

endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/mlaction/rag_sp_token}}",  # <scope>/<secret> that contains an access token
                "DATABRICKS_HOST": "{{secrets/mlaction/rag_sp_host}}", 
            }
        )
    ]
)

# COMMAND ----------


import os 
# url used to send the request to your model from the serverless endpoint
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
db_token= dbutils.secrets.get("mlaction", "rag_sp_token")
os.environ['DATABRICKS_TOKEN'] = db_token

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)

serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)


# COMMAND ----------

    
# stop this command - if you endpoint is not created it will run before the timeout.
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

question = "How LLM can impact US Labour Market?"

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0])
