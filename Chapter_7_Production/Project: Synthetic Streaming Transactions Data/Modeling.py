# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Searching for Signal
# MAGIC
# MAGIC ## Synthetic data - Creating a training set

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating the training set
# MAGIC
# MAGIC With the timeserires features, the first values of the features will be later than the initial raw transactions. We start with determining the earliest raw transaction we want to include in the training set. For the on-demand UDF, nulls will throw an ugly error. For the transaction count, we simply won't have the feature value. Therefore we must have a value for the max product, but not necessarily for the transaction count. We will make sure both have values.

# COMMAND ----------

display(sql("SELECT MIN(lookupTimestamp) as ts FROM product_3minute_max_price_ft"))

# COMMAND ----------

display(sql("SELECT MIN(eventTimestamp) as ts FROM transaction_count_history"))

# COMMAND ----------

# DBTITLE 1,Define the feature lookups
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
fe = FeatureEngineeringClient()

training_feature_lookups = [
    FeatureLookup(
      table_name="transaction_count_history",
      rename_outputs={
          "eventTimestamp": "TransactionTimestamp"
        },
      lookup_key=["CustomerID"],
      feature_names=["transactionCount", "isTimeout"],
      timestamp_lookup_key = "TransactionTimestamp"
    ),
    FeatureLookup(
      table_name="product_3minute_max_price_ft",
      rename_outputs={
          "LookupTimestamp": "TransactionTimestamp"
        },
      lookup_key=['Product'],
      
      timestamp_lookup_key='TransactionTimestamp'
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"MaxProductAmount", "transaction_amount":"Amount"},
      output_name="MaxDifferenceRatio"
    )
]

# COMMAND ----------

# DBTITLE 1,Create the training set
raw_transactions_df = sql("SELECT * FROM raw_transactions WHERE timestamp(TransactionTimestamp) > timestamp('2023-12-12T23:42:54.645+00:00')")

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
    exclude_columns="_rescued_data"
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------

#we may or may not do this
training_df.write.mode("overwrite").saveAsTable("training_data_snapshot")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Training & Registering the model

# COMMAND ----------

training_df = spark.table("training_data")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StringIndexerModel

sidxr = StringIndexer(inputCol="Product", outputCol="ProductIndex")
sidxr_model = sidxr.fit(training_df)
display(sidxr_model.transform(training_df))



# COMMAND ----------

# DBTITLE 1,https://docs.databricks.com/en/_extras/notebooks/source/machine-learning/on-demand-restaurant-recommendation-demo-dynamodb.html
class IsClose(mlflow.pyfunc.PythonModel):
    def predict(self, ctx, inp):
        return (inp['distance'] < 2.5).values

model_name = "fs_packaged_model"
mlflow.set_registry_uri("databricks-uc")

fs.log_model(
    IsClose(),
    model_name,
    flavor=mlflow.pyfunc,
    training_set=training_set,
    registered_model_name=registered_model_name
)

schema = StructType([
    StructField("restaurant_id", IntegerType(), True),
    StructField("json_blob", StringType(), True),
    StructField("ts", TimestampType(), False),
])
data = [
  (2, '{"user_x_coord": 37.79122896768446, "user_y_coord": -122.39362610820227}', datetime(2023, 9, 26, 12, 0, 0)), 
]

scoring_df = spark.createDataFrame(data, schema)

result = fs.score_batch( 
  model_uri = f"models:/{registered_model_name}/1",
  df = scoring_df,
  result_type = 'bool'
)

display(result)

# COMMAND ----------

# DBTITLE 1,Train the model
features_and_label = training_df.columns
training_data = training_df.toPandas()[features_and_label]
 
# X_train = training_data.drop(["Label"], axis=1)
# y_train = training_data.Label.astype(int)
 
# import lightgbm as lgb
# import mlflow.lightgbm
# from mlflow.models.signature import infer_signature
 
# mlflow.lightgbm.autolog()
 
# model = lgb.train(
#   {"num_leaves": 32, "objective": "binary"}, 
#   lgb.Dataset(X_train, label=y_train.values),
#   5
# )

# COMMAND ----------

# DBTITLE 1,Define the feature lookups and training set for the inference model
inference_feature_lookups = [
    FeatureLookup(
      table_name="transaction_count_ft",
      lookup_key="CustomerID",
      feature_names=["transactionCount", "isTimeout"]
    ),
    FeatureLookup(
      table_name="product_3hour_max_price_ft",
      rename_outputs={
          "LookupTimestamp": "TransactionTimestamp"
        },
      lookup_key=['Product'],
      timestamp_lookup_key='TransactionTimestamp'
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"MaxProductAmount", "transaction_amount":"Amount"},
      output_name="MaxDifferenceRatio"
    )
]

raw_transactions_inf_df = sql("SELECT * FROM raw_transactions ORDER BY TransactionTimestamp DESC LIMIT 1000")
inf_training_set = fe.create_training_set(
    df=raw_transactions_inf_df,
    feature_lookups=inference_feature_lookups,
    label="Label",
)

# COMMAND ----------

# DBTITLE 1,Log the model
# Register the model in Model Registry.
# When you use `log_model`, the model is packaged with feature metadata so it automatically looks up features from Feature Store at inference.
fs.log_model(
  model,
  artifact_path="model_packaged",
  flavor=mlflow.lightgbm,
  training_set=inf_training_set,
  registered_model_name="model_name"
)

# COMMAND ----------

from mlflow.tracking import MlflowClient
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
      version_int = int(mv.version)
      if version_int > latest_version:
        latest_version = version_int
    return latest_version

# COMMAND ----------

scored = fs.score_batch(
  f"models:/{model_name}/{get_latest_model_version(model_name)}",
  test_labels,
  result_type="float",
)

# COMMAND ----------

from pyspark.sql.types import BooleanType
 
classify_udf = udf(lambda pred: pred > 0.5, BooleanType())
class_scored = scored.withColumn("person_prediction", classify_udf(scored.prediction))
 
display(class_scored.limit(5))
