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

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# DBTITLE 1,Import & Initialize the DFE Client
import mlflow
mlflow.set_registry_uri("databricks-uc")

model_name = "store_sales_forecasting"
#replace with your model uri runs:/..../model
mlflow.register_model("runs:/7409e084ff0e49ee8d32a223462e4212/model", f"{catalog}.{database_name}.{model_name}")
