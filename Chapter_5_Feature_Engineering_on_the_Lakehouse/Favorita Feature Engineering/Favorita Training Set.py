# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ## Favorita Forecasting - Build a Training Set
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Import & Initialize the DFE Client
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.pandas as ps
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

fe = FeatureEngineeringClient()
training_features = "training_dataset"
raw_data_table = "train_set"
label_name = "sales"

# COMMAND ----------

# DBTITLE 1,Quick reminder what the data looks like
raw_data = sql(f"SELECT * FROM {raw_data_table}")

display(raw_data.take(10))

# COMMAND ----------

# DBTITLE 1,Create training set using feature tables
model_feature_lookups = [
    FeatureLookup(
      table_name="lakehouse_in_action.favorita_forecasting.oil_10d_lag_ft",
      lookup_key="date",
      feature_names="lag10_oil_price"
    ),
    FeatureLookup(
      table_name="lakehouse_in_action.favorita_forecasting.store_holidays_ft",
      lookup_key=["date","store_nbr"]
    ),
    FeatureLookup(
      table_name="lakehouse_in_action.favorita_forecasting.stores_ft",
      lookup_key="store_nbr"
    )  
]

training_set = fe.create_training_set(
    df=raw_data,
    feature_lookups=model_feature_lookups,
    label=label_name
)

training_df = training_set.load_df()
training_df.write.mode("overwrite").saveAsTable("training_set")

# COMMAND ----------

display(training_df)

# COMMAND ----------

# DBTITLE 1,Create the timeseries split for data
training_pd = training_df.toPandas()
dates = np.sort(training_pd["date"].unique())
max_train = len(dates) - 10
tscv = TimeSeriesSplit(test_size=10, max_train_size=max_train)

# COMMAND ----------

# DBTITLE 1,Create the folds for training with test and train data
for i, (train_index, test_index) in enumerate(tscv.split(dates)):
  print(f"Fold {i}:")
  training_dates = dates[train_index]
  testing_dates = dates[test_index]
  min_train_date = np.datetime_as_string(training_dates[0],unit='D')
  max_train_date = np.datetime_as_string(training_dates[-1],unit='D')
  min_test_date = np.datetime_as_string(testing_dates[0],unit='D')
  max_test_date = np.datetime_as_string(testing_dates[-1],unit='D')
  print(f"Training Dates = {min_train_date} to {max_train_date}")
  print(f"Testing Dates = {min_test_date} to {max_test_date}")
  fold_train = training_pd.query(f"date <= '{max_train_date}'")
  X_train = fold_train.drop(label_name, axis=1)
  y_train = fold_train[label_name]
  fold_test = training_pd.query(f"date >= '{min_test_date}' & date <= '{max_test_date}'")
  X_test = fold_test.drop(label_name, axis=1)
  y_test = fold_test[label_name]



# COMMAND ----------


