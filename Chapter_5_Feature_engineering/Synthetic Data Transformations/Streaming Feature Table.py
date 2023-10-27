# Databricks notebook source
# MAGIC %run ../../global-setup $project_name=synthetic_transactions 

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE synthetic_streaming_features ALTER COLUMN CustomerID SET NOT NULL;
# MAGIC ALTER TABLE synthetic_streaming_features ADD CONSTRAINT CustomerID PRIMARY KEY(CustomerID);
