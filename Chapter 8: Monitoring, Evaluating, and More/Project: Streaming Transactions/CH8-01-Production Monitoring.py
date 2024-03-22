# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 8: Monitoring, Evaluating, and More
# MAGIC
# MAGIC ## Creating Monitors for drift and inference performance

# COMMAND ----------

# MAGIC %pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.6-py3-none-any.whl"
# MAGIC

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


