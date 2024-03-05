# Databricks notebook source
# DBTITLE 1,Passed variables via Widgets
# RUN TIME ARGUMENTS
dbutils.widgets.text("env", "dev", "Environment")

#ignored if db is set (we force the databse to the given value in this case)
dbutils.widgets.text("project_name", "", "Project Name")

#Empty value will be set to a database scoped to the current user using project_name
dbutils.widgets.text("db", "", "Database")

# COMMAND ----------

# DBTITLE 1,Checking for compatibility
from pyspark.sql.types import *
import re
import os

# REQUIRES A PROJECT NAME -------------------------
project_name = dbutils.widgets.get('project_name')
possible_projects = ["synthetic_transactions", "favorita_forecasting", "rag_chatbot", "cv_clf"]
assert len(project_name) > 0, "project_name is a required variable"
assert project_name in possible_projects, "project_name unknown, did you type correctly? You can add new projects to the list in the setup file."


# VERIFY DATABRICKS VERSION COMPATIBILITY ----------
min_required_version = "14.0"
version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search('^([0-9]*\.[0-9]*)', version_tag)
assert version_search, f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(min_required_version), f'The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}'


# COMMAND ----------

# DBTITLE 1,Setting a default catalog and project specific database
# DATABASE SETUP -----------------------------------
# Define a current user
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

# Set the UC catalog based on the isolation environment
env = dbutils.widgets.get("env")
if env=="dev":
  catalog = "ml_in_action"
elif env=="prod":
  catalog = "ml_in_prod"  

# Set the database
db = dbutils.widgets.get("db")
if len(db)==0:
  database_name = project_name.replace('-','_')
else:
  database_name = db

def use_and_create_db(catalog, database_name):
  spark.sql(f"""CREATE CATALOG if not exists `{catalog}` """)
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""CREATE DATABASE if not exists `{database_name}` """)

use_and_create_db(catalog, database_name)

print(f"using catalog.database_name `{catalog}`.`{database_name}`")

# With parallel execution this can fail the time of the initialization. Add a few retries to fix these issues
for i in range(10):
  try:
    spark.sql(f"""USE `{catalog}`.`{database_name}`""")
    break
  except Exception as e:
    time.sleep(1)
    if i >= 9:
      raise e

# COMMAND ----------

# DBTITLE 1,Granting permissions
# Granting UC permissions to account users - change if you want your data private    
spark.sql(f"GRANT CREATE, SELECT, USAGE on SCHEMA {catalog}.{database_name} TO `account users`")

# COMMAND ----------

# DBTITLE 1,Setting up volumes
sql(f"""CREATE VOLUME IF NOT EXISTS {catalog}.{database_name}.files""")
volume_file_path = f"/Volumes/{catalog}/{database_name}/files/"
print(f"use volume_file_path {volume_file_path}")

sql(f"""CREATE VOLUME IF NOT EXISTS {catalog}.{database_name}.models""")
volume_model_path = f"/Volumes/{catalog}/{database_name}/models/"
print(f"use volume_model_path {volume_model_path}")

# COMMAND ----------

# DBTITLE 1,Kaggle setup
# Options:
# 1. Comment out this code. Add your kaggle.json file to the same folder as the notebook you ar trying to run. This will require you to move it or make multiple copies.
# 2. Replace with your own credential here.
# 3. [RECCOMENDED] Use the secret scope with your credential.
# 4. Use opendatasets so you can paste in your when prompted. 

import os

os.environ["KAGGLE_USERNAME"] = dbutils.secrets.get("machine-learning-in-action", "kaggle_username")
os.environ["KAGGLE_KEY"] = dbutils.secrets.get("machine-learning-in-action", "kaggle_key")

# COMMAND ----------

# DBTITLE 1,Code to copy-paste for setting up the CLI
# curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# /usr/local/bin/databricks -v

# /usr/local/bin/databricks configure

# /usr/local/bin/databricks secrets list-secrets "machine-learning-in-action"

# /usr/local/bin/databricks secrets create-scope "machine-learning-in-action"

# /usr/local/bin/databricks secrets put-secret --json '{"scope": "machine-learning-in-action","key": "kaggle_username","string_value": "readers-username"}'
# /usr/local/bin/databricks secrets put-secret --json '{"scope": "machine-learning-in-action","key": "kaggle_key","string_value": "readers-api-key"}'

# COMMAND ----------

# DBTITLE 1,Specific settings for the Multilabel Image Classification project
if project_name == "cv_clf":
  try:
    print("You are required to set your token and host if you want to use MLflow tracking while using DDP \n")
    # This is needed for later in the notebook
    print("Setting your db_token and db_host \n")
    db_host = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .extraContext()
        .apply("api_url")
    )
    db_token = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    )
  except:
    print("Your MLFLow logging may not function correctly due to the missing db_host and db_token variables \n")


if project_name == "cv_clf": 
  print("We are setting important repeatable paths for you! Consult global-setup if necessary to change. \n")
  
  main_dir_uc = f"/Volumes/{catalog}/{database_name}/files/intel_image_clf/raw_images"
  main_dir_2write = f"/Volumes/{catalog}/{database_name}/files/intel_image_clf/"

  print(f"Your main_dir_uc Volumes is set to :{main_dir_uc}.")
  print(f"Your main_dir_2write is set to :{main_dir_2write}.")

  try:  
    data_dir_Train = f"{main_dir_uc}/seg_train"
    data_dir_Test = f"{main_dir_uc}/seg_test"
    data_dir_pred = f"{main_dir_uc}/seg_pred/seg_pred"

    print("We are setting your train_dir, valid_dir and pred_files.")
    train_dir = data_dir_Train + "/seg_train"
    valid_dir = data_dir_Test + "/seg_test"
    pred_files = [os.path.join(data_dir_pred, f) for f in os.listdir(data_dir_pred)]

    labels_dict_train = {f"{f}":len(os.listdir(os.path.join(train_dir, f))) for f in os.listdir(train_dir)}
    labels_dict_valid = {f"{f}":len(os.listdir(os.path.join(valid_dir, f))) for f in os.listdir(valid_dir)}
    
    outcomes = os.listdir(train_dir)
   
    print(f"Labels we are working with: {outcomes}.")

    train_delta_path =f"{main_dir_2write}train_imgs_main.delta"
    val_delta_path = f"{main_dir_2write}valid_imgs_main.delta"
    
    print(f"Your train_delta_path is set to: {train_delta_path}")
    print(f"Your val_delta_path is set to: {val_delta_path}")

  except: 
    print("Verify you have downloaded your images and extracted them under Volumes. \n")

# COMMAND ----------

# Temporary as we need routing to be in sdk
class EndpointApiClient:
    def __init__(self):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_inference_endpoint(self, endpoint_name, served_models, auto_capture_config = None):
        data = {"name": endpoint_name, "config": {"served_models": served_models, "auto_capture_config": auto_capture_config}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}", allow_error=True)
      
      
    def inference_endpoint_exists(self, endpoint_name):
      ep = self.get_inference_endpoint(endpoint_name)
      if 'error_code' in ep and ep['error_code'] == 'RESOURCE_DOES_NOT_EXIST':
          return False
      if 'error_code' in ep and ep['error_code'] != 'RESOURCE_DOES_NOT_EXIST':
          raise Exception(f"enpoint exists ? {ep}")
      return True

    def create_endpoint_if_not_exists(self, endpoint_name, model_name, model_version, workload_size, scale_to_zero_enabled=True, wait_start=True, auto_capture_config = None, environment_vars = {}):
      models = [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
            "environment_vars": environment_vars
      }]
      if not self.inference_endpoint_exists(endpoint_name):
        r = self.create_inference_endpoint(endpoint_name, models, auto_capture_config)
      #Make sure we have the proper version deployed
      else:
        ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            self.wait_endpoint_start(endpoint_name)
            ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            model_deployed = ep['pending_config']['served_models'][0]
            print(f"Error with the model deployed: {model_deployed} - state {ep['state']}")
        else:
            model_deployed = ep['config']['served_models'][0]
        if model_deployed['model_version'] != model_version:
          print(f"Current model is version {model_deployed['model_version']}. Updating to {model_version}...")
          u = self.update_model_endpoint(endpoint_name, {"served_models": models})
      if wait_start:
        self.wait_endpoint_start(endpoint_name)
      
      
    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_model_endpoint(self, endpoint_name, conf):
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", conf)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def wait_endpoint_start(self, endpoint_name):
      i = 0
      while self.get_inference_endpoint(endpoint_name)['state']['config_update'] == "IN_PROGRESS" and i < 500:
        if i % 10 == 0:
          print("waiting for endpoint to build model image and start...")
        time.sleep(10)
        i += 1
      ep = self.get_inference_endpoint(endpoint_name)
      if ep['state'].get("ready", None) != "READY":
        print(f"Error creating the endpoint: {ep}")
        
      
    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri, data = {}, allow_error = False):
        r = requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers)
        return self._process(r, allow_error)

    def _post(self, uri, data = {}, allow_error = False):
        return self._process(requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _put(self, uri, data = {}, allow_error = False):
        return self._process(requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _delete(self, uri, data = {}, allow_error = False):
        return self._process(requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _process(self, r, allow_error = False):
      if r.status_code == 500 or r.status_code == 403 or not allow_error:
        print(r.text)
        r.raise_for_status()
      return r.json()
   

