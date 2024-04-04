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
mlflow.register_model("runs:/737b064284b640d38829f71accf3d022/model", f"{catalog}.{database_name}.{model_name}")
