# Databricks notebook source
# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------


def cleanup(table_name):
  dbutils.fs.rm(f"{volume_file_path}/{table_name}", True)
  sql(f"DROP TABLE IF EXISTS {table_name}")

# COMMAND ----------

cleanup("prod_transactions")

# COMMAND ----------

cleanup("transaction_count_ft")

# COMMAND ----------

cleanup("transaction_count_history")
