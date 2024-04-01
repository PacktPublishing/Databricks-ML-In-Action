# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Productionizing ML on Databricks
# MAGIC
# MAGIC ## Wrapping and Logging the Baseline Model
# MAGIC

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC
# MAGIC %pip install --upgrade scikit-learn==1.4.0rc1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

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

table_name = "ml_in_action.synthetic_transactions.raw_transactions"
ft_name = "product_3minute_max_price_ft"

if not spark.catalog.tableExists(ft_name) or spark.table(tableName=ft_name).isEmpty():
  print("problem")
else:  
  raw_transactions_df = sql(
    f"""
    SELECT rt.* FROM {table_name} rt 
    INNER JOIN (SELECT MIN(LookupTimestamp) as min_timestamp FROM {ft_name}) ts ON rt.TransactionTimestamp >= (ts.min_timestamp)
    """)

# COMMAND ----------

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
    exclude_columns="_rescued_data"
)

# COMMAND ----------

display(training_set.load_df())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Training & Registering the model

# COMMAND ----------

import pandas as pd
import mlflow
mlflow.set_registry_uri("databricks-uc")

model_name = "packaged_transaction_model"
full_model_name = f'{catalog}.{database_name}.{model_name}'
model_description = "MLflow custom python function wrapper around a LightGBM model with embedded pre-processing. The wrapper provides data preprocessing so that the model can be applied to input dataframe directly without training/serving skew. This model serves to classify transactions as 0/1 for learning purposes."

model_artifact_path = volume_model_path +  model_name
dbutils.fs.mkdirs(model_artifact_path)

# COMMAND ----------

!pip freeze > /Volumes/ml_in_prod/synthetic_transactions/models/packaged_transaction_model/requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a model wrapper

# COMMAND ----------

# DBTITLE 1,PyFunc Wrapper for Transaction Model
class TransactionModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, _clf, X, y, numeric_columns, cat_columns):
    ## Train test split
    nrows = len(X)
    split_row = round(.8*nrows)
    self.X_train, self.X_test, self.y_train, self.y_test = X[:split_row],X[split_row:],y[:split_row],y[split_row:]
    self.numeric_columns = numeric_columns
    self.cat_columns = cat_columns

    ## OneHot Encoding
    from sklearn.preprocessing import OneHotEncoder  
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist').set_output(transform="pandas")
    self.encoder = ohe.fit(self.X_train[self.cat_columns])

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

    self.model = _clf.fit(self.X_train_processed, self.y_train)

    def _evaluation_metrics(model, X, y):
      from sklearn.metrics import log_loss
      y_pred = model.predict(X)
      log_loss = log_loss(y, y_pred)
      return log_loss
      
    self.log_loss = _evaluation_metrics(model=self.model, X=self.X_test_processed, y=self.y_test)
    
    def _model_signature(model, X):
      from mlflow.models import infer_signature
      y_preds=model.predict(X)
      return infer_signature(X, y_preds)
  
    self.model_signature = _model_signature(model=self.model, X=self.X_test_processed)

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


# COMMAND ----------

# MAGIC %md
# MAGIC #### Train, log, and register the model

# COMMAND ----------

experiment_name = model_artifact_path
experiment = mlflow.get_experiment_by_name(experiment_name)
if not experiment:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
else:
    experiment_id = experiment.experiment_id

mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False,
    disable=False,
    exclusive=False
)

with mlflow.start_run(experiment_id = experiment_id ) as run:
  mlflow.log_params({'Input-table-location': "ml_in_action.synthetic_transactions.raw_transactions",
                    'Training-feature-lookups': training_feature_lookups})
  
  training_data = training_set.load_df().toPandas()
  X = training_data.drop(["Label"], axis=1)
  y = training_data.Label.astype(int)
  
  #columns we want to scale
  numeric_columns = []
  #columns we want to factorize
  cat_columns = ["Product","CustomerID"]
  
  import lightgbm as lgb
  myLGBM = TransactionModelWrapper(_clf = lgb.LGBMClassifier(), X=X, y=y, numeric_columns = numeric_columns,cat_columns = cat_columns)

  eval_data = myLGBM.X_test_processed
  eval_data["Label"] = myLGBM.y_test
  model_info = mlflow.sklearn.log_model(myLGBM.model, "lgbm_model", signature=myLGBM.model_signature,extra_pip_requirements=f"{model_artifact_path}/requirements.txt")
  result = mlflow.evaluate(
       model_info.model_uri,
       eval_data,
       targets="Label",
       model_type="classifier"
   )

  ##------- log pyfunc custom model -------##
  
  fe.log_model(registered_model_name=model_name, model=myLGBM, flavor=mlflow.pyfunc, training_set=training_set, artifact_path="model_package", infer_input_example=X)



# COMMAND ----------

# MAGIC %md
# MAGIC ####Update the data for the registered model

# COMMAND ----------

from mlia_utils.mlflow_funcs import get_latest_model_version
mlfclient = mlflow.tracking.MlflowClient()

model_details = mlfclient.get_registered_model(model_name)
if model_details.description == "":
  mlfclient.update_registered_model(
    name=full_model_name,
    description=model_description
    )

model_version = get_latest_model_version(full_model_name)
mlfclient.set_model_version_tag(name=full_model_name, key="validation_status", value="needs_tested", version=str(model_version))
mlfclient.set_model_version_tag(name=full_model_name, key="project", value=project_name, version=str(model_version))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Test ability to predict

# COMMAND ----------

myLGBM.predict(spark, X)

# COMMAND ----------


