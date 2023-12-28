# Databricks notebook source
# DBTITLE 1,Passed variables via Widgets
# RUN TIME ARGUMENTS
# Minimum Databricks Runtime version allowed for notebooks attaching to a cluster
dbutils.widgets.text("min_dbr_version", "13.0", "Min required DBR version")

dbutils.widgets.text("catalog", "ml_in_action", "Catalog")

#ignored if db is set (we force the databse to the given value in this case)
dbutils.widgets.text("project_name", "", "Project Name")

#Empty value will be set to a database scoped to the current user using project_name
dbutils.widgets.text("db", "", "Database")

# COMMAND ----------

# DBTITLE 1,Running checks
from pyspark.sql.types import *
import re

# REQUIRES A PROJECT NAME -------------------------
project_name = dbutils.widgets.get('project_name')
possible_projects = ["synthetic_transactions", "favorita_forecasting", "rag_chatbot", "cv_clf"]
assert len(project_name) > 0, "project_name is a required variable"
assert project_name in possible_projects, "project_name unknown, did you type correctly? You can add new projects to the list in the setup file."


# VERIFY DATABRICKS VERSION COMPATIBILITY ----------
try:
  min_required_version = dbutils.widgets.get("min_dbr_version")
except:
  min_required_version = "13.0"

if project_name in ["rag_chatbot", "cv_clf"]:
  min_required_version == "14.0"

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

# COMMAND ----------

# DBTITLE 1,Get Kaggle credentials using secrets
# import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
# os.environ['kaggle_username'] = dbutils.secrets.get("lakehouse-in-action", "kaggle_username")

# # os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
# os.environ['kaggle_key'] = dbutils.secrets.get("lakehouse-in-action", "kaggle_key")

# COMMAND ----------

# import os

# os.environ['kaggle_username'] = dbutils.secrets.get("lakehouse-in-action", "kaggle_username")
# os.environ['kaggle_key'] = dbutils.secrets.get("lakehouse-in-action", "kaggle_key")
