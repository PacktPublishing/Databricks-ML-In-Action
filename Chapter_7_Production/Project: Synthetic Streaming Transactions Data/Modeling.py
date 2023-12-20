# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Production ML
# MAGIC
# MAGIC ## Synthetic data - Creating a training set

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

model_name = "packaged_transaction_model"
model_artifact_path = volume_file_path + "/" + model_name
stringIndexerPath = model_artifact_path + "/string-indexer"
#indexToStringPath = model_artifact_path + "/index-to-string"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Training & Registering the model

# COMMAND ----------

training_df = spark.table("training_data_snapshot")

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StringIndexerModel, IndexToString

si = StringIndexer(inputCol="Product", outputCol="ProductIndex")
si_model = si.fit(training_df)
si_model.save(stringIndexerPath)
#display(si_model.transform(training_df))
training_df = si_model.transform(training_df)

# COMMAND ----------


features_and_label = training_df.columns
training_data = training_df.toPandas()[features_and_label]
 
X_train = training_data.drop(["person"], axis=1)
y_train = training_data.person.astype(int)
 
import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
 
mlflow.lightgbm.autolog()
 
model = lgb.train(
  {"num_leaves": 32, "objective": "binary"}, 
  lgb.Dataset(X_train, label=y_train.values),
  5
)

# COMMAND ----------

# DBTITLE 1,PyFunc Wrapper for Transaction Model
class BinaryClassificationModel(mlflow.pyfunc.PythonModel):
  def etl(self, )
  def predict(self, ctx, inp):
      return (inp['distance'] < 2.5).values



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

raw_transactions_inf_df = sql("SELECT * FROM raw_transactions ORDER BY TransactionTimestamp DESC LIMIT 1")
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

# COMMAND ----------

https://docs.databricks.com/en/_extras/notebooks/source/machine-learning/on-demand-restaurant-recommendation-demo-dynamodb.html 
