# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Searching for Signal
# MAGIC
# MAGIC ## Favorita Forecasting -Favorita Advanced Experiment Tracking
# MAGIC
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

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

# DBTITLE 1,Quick reminder what the data looks like
raw_data = sql(f"SELECT * FROM {raw_data_table}")

display(raw_data.take(10))

# COMMAND ----------

# DBTITLE 1,Create a list of FeatureLookups
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
      lookup_key="store_nbr",
      feature_names=["cluster","store_type"]
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

display(training_df)

# COMMAND ----------

automl_data = training_df.filter("date > '2016-12-31'")

summary = databricks.automl.regress(automl_data, 
                                    target_col=label_name,
                                    time_col="date",
                                    timeout_minutes=10,
                                    exclude_cols=['id']
                                    )

# COMMAND ----------

from pandas import Timestamp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.sklearn import DatetimeImputer
from databricks.automl_runtime.sklearn import OneHotEncoder
from databricks.automl_runtime.sklearn import TimestampTransformer
from sklearn.preprocessing import StandardScaler

imputers = {
  "date": DatetimeImputer(),
}

datetime_transformers = []

for col in ["date"]:
    ohe_transformer = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse=False, handle_unknown="indicator"), [TimestampTransformer.HOUR_COLUMN_INDEX])],
        remainder="passthrough")
    timestamp_preprocessor = Pipeline([
        (f"impute_{col}", imputers[col]),
        (f"transform_{col}", TimestampTransformer()),
        (f"onehot_encode_{col}", ohe_transformer),
        (f"standardize_{col}", StandardScaler()),
    ])
    datetime_transformers.append((f"timestamp_{col}", timestamp_preprocessor, [col]))

# COMMAND ----------

dates = np.sort(training_pd["date"].unique())
tscv = TimeSeriesSplit(test_size=90, n_splits=3)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "random_state": 14,
    "criterion": "squared_error",
}


# COMMAND ----------

window_lengths = [10,30,60,90]

with mlflow.start_run(run_name="Forecast Window Length") as run:
  for fold_index, (train_index, test_index) in enumerate(tscv.split(dates)):
    print(f"Fold {fold_index}:")
    ## Training
    training_dates = dates[train_index]
    min_train_date = np.datetime_as_string(training_dates[0],unit='D')
    max_train_date = np.datetime_as_string(training_dates[-1],unit='D')
    print(f"Training Dates = {min_train_date} to {max_train_date}")
    fold_train = training_pd.query(f"date <= '{max_train_date}'")
    X_train = fold_train.drop(label_name, axis=1).drop("date", axis=1)
    y_train = fold_train[label_name]
    model = RandomForestRegressor(**params)
    model.fit(X_train,y_train)

    ## Testing
    testing_dates = dates[test_index]
    min_test_date = np.datetime_as_string(testing_dates[0],unit='D')
    test_score = {}
    for w in window_lengths:
      max_test_date = np.datetime_as_string(testing_dates[w-1],unit='D')
      print(f"Testing Dates = {min_test_date} to {max_test_date}")
      fold_test = training_pd.query(f"date >= '{min_test_date}' & date <= '{max_test_date}'")
      X_test = fold_test.drop(label_name, axis=1).drop("date", axis=1)
      y_test = fold_test[label_name]

      with mlflow.start_run(run_name=f"Fold {fold_index}, window {w}", nested=True):
        mlflow.log_param("run_dates", f"Fold {fold_index}, window {w} - Training Dates = {min_train_date} to {max_train_date}, Testing Dates = {min_test_date} to {max_test_date}")
        mse = mean_squared_error(y_test, model.predict(X_test))
        print("The mean squared error (MSE) on test set: {:.2f}".format(mse))

        ## Generate MSE
        test_score[w] = np.zeros((params["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(model.staged_predict(X_test)):
            test_score[w][i] = mean_squared_error(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Squared Error")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        model.train_score_,
        "b-",
        label="Training Set Error",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score[w], "r-", label="Test Set Error"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Squared Error")
    fig.tight_layout()
    ## Log figure
    mlflow.log_figure(fig, f"fold_{fold_index}_mean_square_error.png")            

          



# COMMAND ----------



