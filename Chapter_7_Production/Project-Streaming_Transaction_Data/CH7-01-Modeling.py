# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Production ML
# MAGIC
# MAGIC ## Synthetic data - Modeling
# MAGIC

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC
# MAGIC %pip install --upgrade scikit-learn==1.4.0rc1

# COMMAND ----------

dbutils.library.restartPython()

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
    table_name="prod_transaction_count_history",
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

raw_transactions_df = sql("SELECT * FROM prod_raw_transactions WHERE timestamp(TransactionTimestamp) > timestamp('2023-12-12T23:42:54.645+00:00')")

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
    exclude_columns="_rescued_data"
)

# COMMAND ----------

display(training_set.load_df())

# COMMAND ----------

#columns we want to scale
numeric_columns = []
#columns we want to factorize
cat_columns = ["Product","CustomerID"]

# COMMAND ----------

# MAGIC
# MAGIC %md 
# MAGIC ### Creating an inference training set

# COMMAND ----------

inference_feature_lookups = [
  FeatureLookup(
    table_name="prod_transaction_count_ft",
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

inf_transactions_df = sql("SELECT * FROM prod_raw_transactions ORDER BY  TransactionTimestamp DESC LIMIT 1")

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


import pandas as pd
import mlflow
mlflow.set_registry_uri("databricks-uc")

model_name = "packaged_transaction_model"
model_artifact_path = volume_model_path +  model_name
dbutils.fs.mkdirs(model_artifact_path)

# COMMAND ----------

!pip freeze > /Volumes/ml_in_action/synthetic_transactions/models/packaged_transaction_model/requirements.txt

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
  
  def __init__(self, model, X, y, numeric_columns, cat_columns):
    self.model = model

    ## Train test split
    from sklearn.model_selection import train_test_split
    nrows = len(X)
    split_row = round(.8*nrows)
    self.X_train, self.X_test, self.y_train, self.y_test = X[:split_row],X[split_row:],y[:split_row],y[split_row:]
    self.numeric_columns = numeric_columns
    self.cat_columns = cat_columns

    ## OneHot Encoding
    from sklearn.preprocessing import OneHotEncoder  
    ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    self.encoder = ohe.fit(X_train[self.cat_columns])

    ## Numerical scaling
    from sklearn.preprocessing import StandardScaler 
    #remove comment to add to text in chapter ---> create a scaler for our numeric variables only run this on the training dataset and use to scale test set later.
    scaler = StandardScaler()
    if len(self.numeric_columns):
      self.fitted_scaler = scaler.fit(self.X_train[self.numeric_columns])
    else:
      self.fitted_scaler=None
    
    self.X_train_processed = TransactionModelWrapper.preprocess_data(self.X_train, self.numeric_columns, self.fitted_scaler,self.cat_columns, self.encoder)
    self.X_test_processed  = TransactionModelWrapper.preprocess_data(self.X_test, self.numeric_columns, self.fitted_scaler,self.cat_columns,self.encoder)

    def _evaluation_metrics(model, X, y):
      from sklearn.metrics import log_loss
      y_pred = model.predict(X)
      log_loss = log_loss(y, y_pred)
      return log_loss
      
    self.log_loss = _evaluation_metrics(model=self.model, X=self.X_test_processed, y=self.y_test)
  
  def predict(self, context, input_data: pd.DataFrame)->pd.DataFrame:
    input_processed = TransactionModelWrapper.preprocess_data(df=input_data, numeric_columns=self.numeric_columns, fitted_scaler=self.fitted_scaler ,cat_columns=self.cat_columns, encoder=self.encoder)
    return pd.DataFrame(self.model.predict(input_processed), columns=['predicted'])
  
  @staticmethod
  def preprocess_data(df, numeric_columns,fitted_scaler,cat_columns, encoder):
    if "TransactionTimestamp" in df.columns:
      try:
        df = df.drop("TransactionTimestamp",axis=1)
      except:
        df = df.drop("TransactionTimestamp")
    one_hot_encoded = encoder.transform(df[cat_columns])
    df = pd.concat([df,one_hot_encoded],axis=1).drop(columns=cat_columns)
    df["isTimeout"] = df["isTimeout"].astype('bool')
    ## scale the numeric columns with the pre-built scaler
    if len(numeric_columns):
      ndf = df[numeric_columns].copy()
      df[numeric_columns] = fitted_scaler.transform(ndf[numeric_columns])
    return df
  
  @staticmethod
  def fit(X, y):
    import lightgbm as lgb
    _clf = lgb.LGBMClassifier()
    lgbm_model = _clf.fit(X, y)
    return lgbm_model
  

# COMMAND ----------

from mlflow.models import infer_signature
import json
context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
experiment_name = context['extraContext']['notebook_path']
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False,
    disable=False,
    exclusive=False
)

with mlflow.start_run(experiment_id = experiment_id ) as run:
  mlflow.log_params({'Input-table-location': f"{catalog}.{database_name}.prod_raw_transactions",
                    'Training-feature-lookups': training_feature_lookups, 
                    'Inference-feature-lookups': inference_feature_lookups})
  
  from sklearn.model_selection import train_test_split
  training_data = training_set.load_df().toPandas()
  X = training_data.drop(["Label"], axis=1)
  y = training_data.Label.astype(int)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2024)

  ## OneHotEncoding
  from sklearn.preprocessing import OneHotEncoder  
  ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
  encoder = ohe.fit(X_train[cat_columns])

  from sklearn.preprocessing import StandardScaler 
  # create a scaler for our numeric variables
  # only run this on the training dataset and use to scale test set later.
  scaler = StandardScaler()
  if len(numeric_columns):
    fitted_scaler = scaler.fit(X_train[numeric_columns])
  else:
    fitted_scaler=None
  X_train_processed = TransactionModelWrapper.preprocess_data(df=X_train, numeric_columns=numeric_columns, fitted_scaler=fitted_scaler,cat_columns=cat_columns, encoder=encoder)
  X_test_processed = TransactionModelWrapper.preprocess_data(df=X_test, numeric_columns=numeric_columns, fitted_scaler=fitted_scaler,cat_columns=cat_columns, encoder=encoder)
  
  #Train a model
  lgbm_model = TransactionModelWrapper.fit(X=X_train_processed, y=y_train)
  
  y_preds=lgbm_model.predict(X_test_processed)
  signature = infer_signature(X_test_processed, y_preds)
  eval_data = X_test_processed
  eval_data["Label"] = y_test
  model_info = mlflow.sklearn.log_model(lgbm_model, "lgbm_model", signature=signature)
  result = mlflow.evaluate(
       model_info.model_uri,
       eval_data,
       targets="Label",
       model_type="classifier"
   )

  ##------- log pyfunc custom model -------##
   # make an instance of the Pyfunc Class

  myLGBM = TransactionModelWrapper(model = lgbm_model, X=X, y=y, numeric_columns = numeric_columns,cat_columns = cat_columns)
  
  fe.log_model(registered_model_name=model_name, model=myLGBM, flavor=mlflow.pyfunc, training_set=inferencing_set, artifact_path="model_package")
  
# # programmatically get the latest Run ID
# runs = mlflow.search_runs(mlflow.get_experiment_by_name(experiment_name).experiment_id)
# latest_run_id = runs.sort_values('end_time').iloc[-1]["run_id"]
# print('The latest run id: ', latest_run_id)
