# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC - This notebook performs exploratory data analysis on the dataset.
# MAGIC - To expand on the analysis, attach this notebook to a cluster with runtime version **13.0.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/3068301312211919).

# COMMAND ----------

import os
import uuid
import pandas as pd
import shutil
import databricks.automl_runtime
import pyspark.pandas as ps

import mlflow

ps.options.plotting.backend = "matplotlib"

# Download input data from mlflow into a pyspark.pandas DataFrame
# create temp directory to download data
exp_temp_dir = os.path.join("/dbfs/tmp", str(uuid.uuid4())[:8])
os.makedirs(exp_temp_dir)

# download the artifact and read it
exp_data_path = mlflow.artifacts.download_artifacts(run_id="f155d589aed4483ba3605a3108e36ec6", artifact_path="data", dst_path=exp_temp_dir)
exp_file_path = os.path.join(exp_data_path, "training_data")
exp_file_path  = "file://" + exp_file_path

df = ps.from_pandas(pd.read_parquet(exp_file_path)).spark.cache()

target_col = "aggSales"
time_col = "date"
id_cols = ["store_nbr"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate data

# COMMAND ----------

group_cols = [time_col] + id_cols

df_aggregated = df \
  .groupby(group_cols) \
  .agg(aggSales=(target_col, "avg")) \
  .reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time column Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Show the time range for the time series

# COMMAND ----------

df_time_range = df_aggregated.groupby(id_cols).agg(min=(time_col, "min"), max=(time_col, "max"))
display(df_time_range.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Value Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Time series target value status

# COMMAND ----------

selected_cols = id_cols + [target_col]
target_stats_df = df_aggregated[selected_cols].groupby(id_cols).describe()
display(target_stats_df.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Check the number of missing values in the target column.

# COMMAND ----------

def num_nulls(x):
  num_nulls = x.isnull().sum()
  return pd.Series(num_nulls)

null_stats_df = df_aggregated[selected_cols].groupby(id_cols).apply(num_nulls)[target_col]
display(null_stats_df.to_frame().reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize the Data

# COMMAND ----------

# Select one id from id columns
idx = df_aggregated[id_cols].to_pandas().astype(str).agg('-'.join, axis=1).unique()[0] # change index here to see other identities
idx_list = idx.split("-")
df_sub = df.loc[(df["store_nbr"] == idx_list[0])]

df_sub = df_sub.filter(items=[time_col, target_col])
df_sub.set_index(time_col, inplace=True)
df_sub[target_col] = df_sub[target_col].astype("float")

fig = df_sub.plot()

# COMMAND ----------

# delete the temp data
shutil.rmtree(exp_temp_dir)
