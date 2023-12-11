# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Feature Engineering
# MAGIC
# MAGIC ##Synthetic data - Updating to Feature Tables

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# DBTITLE 1,Constraints using SQL
# MAGIC %sql
# MAGIC ALTER TABLE transaction_count_ft ALTER COLUMN CustomerID SET NOT NULL;
# MAGIC ALTER TABLE transaction_count_ft ADD PRIMARY KEY(CustomerID);
# MAGIC
# MAGIC ALTER TABLE transaction_count_history ALTER COLUMN CustomerID SET NOT NULL;
# MAGIC ALTER TABLE transaction_count_history ALTER COLUMN eventTimestamp SET NOT NULL;
# MAGIC ALTER TABLE transaction_count_history ADD PRIMARY KEY(CustomerID, eventTimestamp TIMESERIES);

# COMMAND ----------

# DBTITLE 1,Tags in Python
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

fe.set_feature_table_tag(name="transaction_count_ft", key="FE_role", value="online_serving")
fe.set_feature_table_tag(name="transaction_count_history", key="FE_role", value="training_data")
