# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Production ML
# MAGIC
# MAGIC ## Synthetic data - Modeling
# MAGIC

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Recreate our training_set

# COMMAND ----------

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

raw_transactions_df = sql("SELECT * FROM raw_transactions WHERE timestamp(TransactionTimestamp) > timestamp('2023-12-12T23:42:54.645+00:00')")

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
    exclude_columns="_rescued_data"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating an inference training set

# COMMAND ----------

inference_feature_lookups = [
  FeatureLookup(
    table_name="transaction_count_ft",
    lookup_key=["CustomerID"],
    feature_names=["transactionCount", "isTimeout"]
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

inf_transactions_df = sql("SELECT * FROM raw_transactions ORDER BY  TransactionTimestamp DESC LIMIT 1")

inferencing_set = fe.create_training_set(
    df=inf_transactions_df,
    feature_lookups=inference_feature_lookups,
    label="Label",
    exclude_columns="_rescued_data"
)

# COMMAND ----------

## We are testing the functionality. We use the display command to force plan execution. Spark using lazy execution. 
display(inferencing_set.load_df())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Training & Registering the model

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

model_name = "packaged_transaction_model"
model_artifact_path = volume_file_path + "/" + model_name
stringIndexerPath = model_artifact_path + "/string-indexer"
#indexToStringPath = model_artifact_path + "/index-to-string"

# COMMAND ----------




# COMMAND ----------

# DBTITLE 1,PyFunc Wrapper for Transaction Model
class TransactionModelWrapper(mlflow.pyfunc.PythonModel):
  '''
  LightGBM with embedded pre-processing.
  
  This class is an MLflow custom python function wrapper around a LGB model.
  The wrapper provides data preprocessing so that the model can be applied to input dataframe directly.
  :Input: to the model is pandas dataframe
  :Output: predicted class for each transaction

  ????The model declares current local versions of XGBoost and pillow as dependencies in its
  conda environment file.  
  '''
  def __init__(self, training_set, inferencing_set, numeric_columns):
    self.training_df = training_set.load_df()

    ## Product name to product index
    from pyspark.ml.feature import StringIndexer, StringIndexerModel, IndexToString
    si = StringIndexer(inputCol="Product", outputCol="ProductIndex")
    self.indexer = si.fit(self.training_df)

    ## Train test split
    import pandas
    training_data = self.indexer.transform(self.training_df)
    self.training_data = training_data.toPandas()
  
    X = training_data.drop(["Product","Label"], axis=1)
    y = training_data.Label.astype(int)

    from sklearn.model_selection import train_test_split
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.30, random_state=2024)
    self.numeric_columns = numeric_columns

    ## Numerical scaling
    from sklearn.preprocessing import StandardScaler 
    #remove comment to add to text in chapter ---> create a scaler for our numeric variables only run this on the training dataset and use to scale test set later.
    scaler = StandardScaler()
    self.fitted_scaler = scaler.fit(self.X_train[self.numeric_columns])
    self.X_train_processed = scale_data(self.X_train, self.numeric_columns, self.fitted_scaler)
    self.X_test_processed  = scale_data(self.X_test, self.numeric_columns, self.fitted_scaler)


  def fit(X, y):
    """
    :return: dictionary with fields 'loss' (scalar loss) and 'model' fitted model instance
    """
    import lightgbm as lgb
    _clf = lgb.LGBMClassifier(boosting_type='gbdt', objective='clf:binary_logloss')
    lgbm_model = _clf.fit(X, y)
    
    from sklearn.model_selection import cross_val_score
    score = cross_val_score(_clf, X, y, scoring='binary_logloss').mean()
    
    return {'mean_training_loss': score, 'model': lgbm_model}
  
  fitmodel = fit(self.X_train_processed,y_train)
  self.model = fitmodel["model"]
  self.training_score = fitmodel["mean_training_loss"]

  def _evaluation_metrics(model, X, y):
    import sklearn
    from sklearn import metrics
    y_pred = model.predict(X)
    log_loss = sklearn.metrics.log_loss(y, y_pred)

    return log_loss
    
  self.log_loss = _accuracy_metrics(model=self.model, X=self.X_test_processed, y=self.y_test )
  
  def predict(self, input_data):
    input_processed = self.preprocess_data(X=input_data, numeric_columns=self.numeric_columns, fitted_scaler=self.fitted_scaler , indexer=self.indexer)
    return pd.DataFrame(self.model.predict(input_processed), columns=['predicted'])
  
  def scale_data(df, numeric_columns,fitted_scaler):
    _df = df[numeric_columns].copy()
    
    ## scale the numeric columns with the pre-built scaler
    df[numeric_columns] = fitted_scaler.transform(_df[numeric_columns])
    
    return df
  
  def preprocess_data(df, numeric_columns,fitted_scaler,indexer):
    import pandas
    df = indexer.transform(df).toPandas()
    df = df.drop("Product", axis = 1)

    _df = df[numeric_columns].copy()
    ## scale the numeric columns with the pre-built scaler
    df[numeric_columns] = fitted_scaler.transform(_df[numeric_columns])
    
    return df





# fs.log_model(
#     IsClose(),
#     model_name,
#     flavor=mlflow.pyfunc,
#     training_set=training_set,
#     registered_model_name=registered_model_name
# )

# schema = StructType([
#     StructField("restaurant_id", IntegerType(), True),
#     StructField("json_blob", StringType(), True),
#     StructField("ts", TimestampType(), False),
# ])
# data = [
#   (2, '{"user_x_coord": 37.79122896768446, "user_y_coord": -122.39362610820227}', datetime(2023, 9, 26, 12, 0, 0)), 
# ]

# scoring_df = spark.createDataFrame(data, schema)

# result = fs.score_batch( 
#   model_uri = f"models:/{registered_model_name}/1",
#   df = scoring_df,
#   result_type = 'bool'
# )

# display(result)

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

from mlia_utils import get_latest_model_version

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

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StringIndexerModel, IndexToString

si = StringIndexer(inputCol="Product", outputCol="ProductIndex")
si_model = si.fit(training_df)
si_model.save(stringIndexerPath)
#display(si_model.transform(training_df))
training_df = si_model.transform(training_df)
