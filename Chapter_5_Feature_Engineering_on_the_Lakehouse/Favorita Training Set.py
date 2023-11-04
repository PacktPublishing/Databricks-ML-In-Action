# Databricks notebook source
# MAGIC %md
# MAGIC ## Favorita Forecasting - Build a Training Set
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Import & Initialize the DFE Client
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

fe = FeatureEngineeringClient()
training_features = "training_features"
raw_data_table = "train_set"
label_name = "sales"

# COMMAND ----------

# DBTITLE 1,Quick reminder what the data looks like
display(sql(f"SELECT * FROM {training_features} LIMIT 5"))

# COMMAND ----------

raw_data = sql(f"SELECT * FROM {raw_data_table}")

# COMMAND ----------

model_feature_lookups = [FeatureLookup(table_name=training_features, lookup_key=["date", "store_nbr","family","onpromotion"])]

training_set = fe.create_training_set(df=raw_data, feature_lookups=model_feature_lookups, label=label_name, timestamp_lookup_key="date")
training_pd = training_set.load_df().toPandas()

# COMMAND ----------



dates = np.sort(training_pd["date"].unique())
max_train = len(dates) - 10
tscv = TimeSeriesSplit(test_size=10, max_train_size=max_train)

# COMMAND ----------

for i, (train_index, test_index) in enumerate(tscv.split(dates)):
  print(f"Fold {i}:")
  training_dates = dates[train_index]
  testing_dates = dates[test_index]
  print(f"Max Training Date = {training_dates[-1]}")
  print(f"Testing Dates = {testing_dates[0],testing_dates[-1]}")
  fold_train = training_pd.query[f'date <= {training_dates[-1]}']
  X_train = fold_train.drop(label_name, axis=1)
  y_train = fold_train[label_name]
  fold_test = training_pd.query[f'date >= {testing_dates[0]} & date <= {testing_dates[-1]}']
  X_test = fold_test.drop(label_name, axis=1)
  y_test = fold_test[label_name]



# COMMAND ----------

str(testing_dates[-1])

# COMMAND ----------

max_train

# COMMAND ----------

# DBTITLE 1,Some dates have more than one national holiday
# Create train and test datasets
X = training_pd.drop(label_name, axis=1)
y = training_pd[label_name]
