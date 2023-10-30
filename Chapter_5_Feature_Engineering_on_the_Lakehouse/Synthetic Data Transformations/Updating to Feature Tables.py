# Databricks notebook source
# MAGIC %run ../../global-setup $project_name=synthetic_transactions 

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE synthetic_streaming_features ALTER COLUMN CustomerID SET NOT NULL;
# MAGIC ALTER TABLE synthetic_streaming_features ADD CONSTRAINT CustomerID_synthetic_streaming_features PRIMARY KEY(CustomerID);

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

fe.set_feature_table_tag(name="synthetic_streaming_features", key="FE_role", value="online_serving")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE synthetic_feature_history ALTER COLUMN CustomerID SET NOT NULL;
# MAGIC ALTER TABLE synthetic_feature_history ALTER COLUMN eventTimestamp SET NOT NULL;
# MAGIC ALTER TABLE synthetic_feature_history ADD CONSTRAINT CustomerID_synthetic_feature_history PRIMARY KEY(CustomerID,eventTimestamp TIMESERIES);

# COMMAND ----------

fe.set_feature_table_tag(name="synthetic_feature_history", key="FE_role", value="training_data")

# COMMAND ----------


