# Databricks notebook source
# DBTITLE 1,Passed variables via Widgets
# RUN TIME ARGUMENTS
dbutils.widgets.text("catalog", "ml_in_action", "Catalog")

#ignored if db is set (we force the databse to the given value in this case)
dbutils.widgets.text("project_name", "", "Project Name")

#Empty value will be set to a database scoped to the current user using project_name
dbutils.widgets.text("db", "", "Database")

# COMMAND ----------

# DBTITLE 1,Running checks
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

# DBTITLE 1,Catalog and database setup
# DATABASE SETUP -----------------------------------
# Define a current user
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

# Set the UC catalog
catalog = dbutils.widgets.get("catalog")

# Set the database
db = dbutils.widgets.get("db")
if len(db)==0:
  database_name = project_name.replace('-','_')
else:
  database_name = db

def use_and_create_db(catalog, database_name):
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{database_name}` """)
  
#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
  use_and_create_db(catalog, database_name)
else:
  catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
  if "ml_in_action" not in catalogs:
    spark.sql("CREATE CATALOG IF NOT EXISTS ml_in_action")
    catalog = "ml_in_action"
    use_and_create_db(catalog, database_name)

print(f"using catalog.database_name `{catalog}`.`{database_name}`")

# With parallel execution this can fail the time of the initialization. add a few retries to fix these issues
for i in range(10):
  try:
    spark.sql(f"""USE `{catalog}`.`{database_name}`""")
    break
  except Exception as e:
    time.sleep(1)
    if i >= 9:
      raise e

# Granting UC permissions to account users - change if you want your data private    
spark.sql(f"GRANT CREATE, SELECT, USAGE on SCHEMA {catalog}.{database_name} TO `account users`")


# COMMAND ----------

# DBTITLE 1,Create the volume
sql(f"""CREATE VOLUME IF NOT EXISTS {catalog}.{database_name}.files""")
volume_file_path = f"/Volumes/{catalog}/{database_name}/files/"
print(f"use volume_file_path {volume_file_path}")
sql(f"""CREATE VOLUME IF NOT EXISTS {catalog}.{database_name}.models""")
volume_model_path = f"/Volumes/{catalog}/{database_name}/models/"
print(f"use volume_model_path {volume_model_path}")




# COMMAND ----------

if project_name == "cv_clf":
  try:
    print("You are required to set your token and host if you want to use MLFlow tracking while using DDP \n")
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

  MAIN_DIR_UC = f"/Volumes/{catalog}/{database_name}/files/intel_image_clf/raw_images"
  MAIN_DIR2Write = f"/Volumes/{catalog}/{database_name}/files/intel_image_clf/"
  print(f"Your Main dir for UC Volumes is :{MAIN_DIR_UC} \n")
  print(f"Your main dir to write is :{MAIN_DIR2Write} \n")
  try:  
    data_dir_Train = f"{MAIN_DIR_UC}/seg_train"
    data_dir_Test = f"{MAIN_DIR_UC}/seg_test"
    data_dir_pred = f"{MAIN_DIR_UC}/seg_pred/seg_pred"

    train_dir = data_dir_Train + "/seg_train"
    valid_dir = data_dir_Test + "/seg_test"
    pred_files = [os.path.join(data_dir_pred, f) for f in os.listdir(data_dir_pred)]

    labels_dict_train = {f"{f}":len(os.listdir(os.path.join(train_dir, f))) for f in os.listdir(train_dir)}
    labels_dict_valid = {f"{f}":len(os.listdir(os.path.join(valid_dir, f))) for f in os.listdir(valid_dir)}

    outcomes = os.listdir(train_dir)
    print(outcomes)

    train_delta_path =f"/Volumes/{catalog}/{database_name}/files/intel_image_clf/train_imgs_main.delta"
    val_delta_path = f"/Volumes/{catalog}/{database_name}/files/intel_image_clf/valid_imgs_main.delta"

  except: 
    print("Verify you have downloaded your images and extracted them under Volumes \n")

