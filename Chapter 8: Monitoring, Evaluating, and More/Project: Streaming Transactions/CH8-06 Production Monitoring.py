# Databricks notebook source
# MAGIC %pip install databricks_lakehouse_monitoring

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

from databricks import lakehouse_monitoring as lm

# COMMAND ----------

lm.create_monitor(
    table_name=f"{catalog}.{database_name}.prod_transactions",
    profile_type=lm.TimeSeries(
        timestamp_col="TransactionTimestamp",
        granularities=["5 minutes","30 minutes","1 hour"]
    ),
    slicing_exprs=["Product","CustomerID"],
    schedule=lm.MonitorCronSchedule(
        quartz_cron_expression="0 0 * * * ?", # schedules a refresh every hour
        timezone_id="MST",
    ),
    output_schema_name=f"{catalog}.{database_name}"
)

# COMMAND ----------

lm.create_monitor(
    table_name=f"{catalog}.{database_name}.packaged_transaction_model_predictions",
    profile_type=lm.InferenceLog(
        timestamp_col="TransactionTimestamp",
        granularities=["5 minutes","30 minutes","1 hour"],
        model_id_col="model_version",
        prediction_col="prediction",
        label_col="actual_label",
        problem_type="classification"
    ),
    slicing_exprs=["Product","CustomerID"],
    schedule=lm.MonitorCronSchedule(
        quartz_cron_expression="0 0 * * * ?", # schedules a refresh every hour
        timezone_id="MST",
    ),
    output_schema_name=f"{catalog}.{database_name}"
)

# COMMAND ----------


