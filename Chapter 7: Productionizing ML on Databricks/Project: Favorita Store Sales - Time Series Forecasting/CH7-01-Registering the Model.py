# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Productionizing ML on Databricks
# MAGIC
# MAGIC ## Favorita Forecasting -Registering the Model
# MAGIC
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pasted code

# COMMAND ----------

# DBTITLE 1,Import & Initialize the DFE Client
import mlflow
catalog = catalog
schema = database_name
model_name = "store_sales_forecasting"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/cdbf797cb40e40b8aab30b4fecdffc0b/model", f"{catalog}.{schema}.{model_name}")
