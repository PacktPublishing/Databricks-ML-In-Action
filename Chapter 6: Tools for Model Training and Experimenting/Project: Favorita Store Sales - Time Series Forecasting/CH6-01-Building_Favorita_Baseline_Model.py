# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Searching for Signal
# MAGIC
# MAGIC ## Favorita Forecasting -Favorita Baseline Model
# MAGIC
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# DBTITLE 1,Import & Initialize the DFE Client
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pyspark.pandas as ps
import numpy as np
import mlflow

fe = FeatureEngineeringClient()
training_features = "training_dataset"
raw_data_table = "train_set"
label_name = "sales"
time_column = "date"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Quick reminder what the data looks like

# COMMAND ----------

raw_data = sql(f"SELECT * FROM {raw_data_table}")

display(raw_data.take(10))

# COMMAND ----------

store_data = spark.table("stores_ft")

display(store_data.take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Create FeatureLookups, a training set, and a baseline model

# COMMAND ----------

# DBTITLE 1,Create a list of FeatureLookups
model_feature_lookups = [
    FeatureLookup(
      table_name="oil_10d_lag_ft",
      lookup_key="date",
      feature_names="lag10_oil_price"
    ),
    FeatureLookup(
      table_name="store_holidays_ft",
      lookup_key=["date","store_nbr"]
    ),
    FeatureLookup(
      table_name="stores_ft",
      lookup_key="store_nbr",
      feature_names=["store_cluster","store_type"]
    ),
]

# COMMAND ----------

# DBTITLE 1,Create the training set
training_set = fe.create_training_set(
    df=raw_data,
    feature_lookups=model_feature_lookups,
    label=label_name,
)
training_df = training_set.load_df()

# COMMAND ----------

# DBTITLE 1,Save as table
training_df.write.mode("overwrite").saveAsTable("training_data")

# COMMAND ----------

display(training_df)

# COMMAND ----------

from databricks import automl
automl_data = training_df.filter("date > '2016-12-31'")

summary = automl.regress(automl_data, 
                          target_col=label_name,
                          time_col="date",
                          timeout_minutes=10,
                          exclude_cols=['id']
                          )
