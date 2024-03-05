# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Productionizing ML on Databricks
# MAGIC
# MAGIC ## Create Model Testing Job
# MAGIC Create a Databricks Job to run the tests on our moel registered in UC

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

import mlflow

token = mlflow.utils.databricks_utils._get_command_context().apiToken().get()
headers = {"Authorization": f"Bearer {token}"}
instance = mlflow.utils.databricks_utils.get_webapp_url()

# COMMAND ----------

import requests

def find_job_id(instance, headers, job_name, offset_limit=1000):
    params = {"offset": 0}
    uri = f"{instance}/api/2.1/jobs/list"
    done = False
    job_id = None
    while not done:
        done = True
        res = requests.get(uri, params=params, headers=headers)
        assert res.status_code == 200, f"Job list not returned; {res.content}"
        
        jobs = res.json().get("jobs", [])
        if len(jobs) > 0:
            for job in jobs:
                if job.get("settings", {}).get("name", None) == job_name:
                    job_id = job.get("job_id", None)
                    break

            # if job_id not found; update the offset and try again
            if job_id is None:
                params["offset"] += len(jobs)
                if params["offset"] < offset_limit:
                    done = False
    
    return job_id

def get_job_parameters(job_name, cluster_id, notebook_path):
    params = {
            "name": job_name,
            "tasks": [{"task_key": "webhook_task", 
                       "existing_cluster_id": cluster_id,
                       "notebook_task": {
                           "notebook_path": notebook_path
                       }
                      }]
        }
    return params

def get_create_parameters(job_name, cluster_id, notebook_path):
    api = "api/2.1/jobs/create"
    return api, get_job_parameters(job_name, cluster_id, notebook_path)

def get_reset_parameters(job_name, cluster_id, notebook_path, job_id):
    api = "api/2.1/jobs/reset"
    params = {"job_id": job_id, "new_settings": get_job_parameters(job_name, cluster_id, notebook_path)}
    return api, params

def get_webhook_job(instance, headers, job_name, cluster_id, notebook_path):
    job_id = find_job_id(instance, headers, job_name)
    if job_id is None:
        api, params = get_create_parameters(job_name, cluster_id, notebook_path)
    else:
        api, params = get_reset_parameters(job_name, cluster_id, notebook_path, job_id)
    
    uri = f"{instance}/{api}"
    res = requests.post(uri, headers=headers, json=params)
    assert res.status_code == 200, f"Expected an HTTP 200 response, received {res.status_code}; {res.content}"
    job_id = res.json().get("job_id", job_id)
    return job_id

# COMMAND ----------

notebook_path = mlflow.utils.databricks_utils.get_notebook_path().replace('CH07-07-Create Model Testing Job', "CH7-06-Model Validation")

prefix = f"{catalog}-model-validation"
job_name = f"{prefix}_webhook-job"

#If you want to create the Job in the UI paste the Job_ID here
job_id = get_webhook_job(instance, 
                         headers, 
                         job_name,
                         spark.conf.get("spark.databricks.clusterUsageTags.clusterId"),
                         notebook_path)

print(f"Job ID:   {job_id}")
print(f"Job name: {job_name}")
