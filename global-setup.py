# Databricks notebook source
# DBTITLE 1,Passed variables via Widgets
# RUN TIME ARGUMENTS
# Minimum Databricks Runtime version allowed for notebooks attaching to a cluster
dbutils.widgets.text("min_dbr_version", "12.0", "Min required DBR version")

# Specifying a value will not have fallback and fail if the catalog can't be used/created
dbutils.widgets.text("catalog", "lakehouse_in_action", "Catalog")

#ignored if db is set (we force the databse to the given value in this case)
dbutils.widgets.text("project_name", "", "Project Name")

#Empty value will be set to a database scoped to the current user using project_name
dbutils.widgets.text("db", "", "Database")

dbutils.widgets.text("data_path", "s3://one-env/lakehouse_ml_in_action/", "Data Path")


# COMMAND ----------

# DBTITLE 1,Running checks
from pyspark.sql.types import *
import re

# REQUIRES A PROJECT NAME -------------------------
project_name = dbutils.widgets.get('project_name')
assert len(project_name) > 0, "project_name is a required variable"


# VERIFY DATABRICKS VERSION COMPATIBILITY ----------
try:
  min_required_version = dbutils.widgets.get("min_dbr_version")
except:
  min_required_version = "13.0"

version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search('^([0-9]*\.[0-9]*)', version_tag)
assert version_search, f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(min_required_version), f'The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}'
assert "ml" in version_tag.lower(), f"The Databricks ML runtime must be used. Current version detected doesn't contain 'ml': {version_tag} "

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

# Define data paths in both formats  
data_path = dbutils.widgets.get('data_path')
if len(data_path) == 0:
  cloud_storage_path = f"/Users/{current_user}/lakehouse_in_action/{project_name}"
else:
  cloud_storage_path = f"{data_path}{project_name}"

try:
  dbutils.fs.ls(cloud_storage_path)
except:
  dbutils.fs.mkdirs(cloud_storage_path)

#Try to use the UC catalog when possible. If not will fallback to hive_metastore
catalog = dbutils.widgets.get("catalog")

db = dbutils.widgets.get("db")
if len(catalog) == 0 or catalog == 'hive_metastore':
  database_name = "lakehouse_in_action"
elif len(db)==0:
  database_name = project_name.replace('-','_')
else:
  database_name = db

def use_and_create_db(catalog, database_name, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  if cloud_storage_path == None or catalog != 'hive_metastore':
    spark.sql(f"""create database if not exists `{database_name}` """)
  else:
    spark.sql(f"""create database if not exists `{database_name}` LOCATION '{cloud_storage_path}/tables' """)

  
#If the catalog is defined, we force it to the given value and throw exception if not.
if catalog == 'hive_metastore':
  use_and_create_db(catalog, database_name)
elif len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
  use_and_create_db(catalog, database_name)
else:
  #otherwise we'll try to set the catalog to lakehouse_in_action and create the database here.
  try:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if len(catalogs) == 1 and catalogs[0] == 'hive_metastore':
      print(f"Not able to use UC, using hive_metastore")
      catalog = "hive_metastore"
    else:
      if "lakehouse_in_action" not in catalogs:
        spark.sql("CREATE CATALOG IF NOT EXISTS lakehouse_in_action")
        catalog = "lakehouse_in_action"
    use_and_create_db(catalog, database_name)
  except Exception as e:
    print(f"error with catalog {e}, not able to use UC. Using hive_metastore.")
    catalog = "hive_metastore"

print(f"using cloud_storage_path {cloud_storage_path}")
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
if catalog != 'hive_metastore':
  try:
    spark.sql(f"GRANT CREATE, SELECT, USAGE on SCHEMA {catalog}.{database_name} TO `account users`")
    spark.sql(f"ALTER SCHEMA {catalog}.{database_name} OWNER TO `account users`")
  except Exception as e:
    print("Couldn't grant access to the database for all users:"+str(e))
  

# COMMAND ----------

# DBTITLE 1,Use volumes for projects using UC
if catalog == 'hive_metastore':
  volume_label = ''
  spark_temp_path = f"dbfs:/Users/{current_user}/lakehouse_in_action/{project_name}"
  print(f"Not using volumes for {project_name}, spark_temp_path = {spark_temp_path}")
elif database_name == 'asl_fingerspelling':
  # Pose coordinates for hand movement.
  LPOSE = [13, 15, 17, 19, 21]
  RPOSE = [14, 16, 18, 20, 22]
  POSE = LPOSE + RPOSE

  X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
  Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
  Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

  FEATURE_COLUMNS = X + Y + Z

  X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
  Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
  Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

  RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
  LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]
  RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
  LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]

  volume_label = "asl"
elif database_name == 'parkinsons_freezing_gait_prediction':
  volume_label = "fog"
elif database_name == 'favorita_forecasting':
  volume_label = "fav"
elif database_name == "sql_bot":
  volume_label = "sqlbot"

# COMMAND ----------

# DBTITLE 1,Create the volume
if volume_label != '':
  sql(f"""CREATE VOLUME IF NOT EXISTS {catalog}.{database_name}.{volume_label}_volume""")
  volume_data_path = f"/Volumes/{catalog}/{database_name}/{volume_label}_volume/"
  print(f"using volume_data_path {volume_data_path}")

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
