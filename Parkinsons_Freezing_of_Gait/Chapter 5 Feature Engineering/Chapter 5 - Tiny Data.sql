-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC ## Feature Engineering for Event Detection
-- MAGIC Your objective is to detect the start and stop of each freezing episode and the occurrence in these series of three types of freezing of gait events: Start Hesitation, Turn, and Walking.

-- COMMAND ----------

-- MAGIC %run ./../setup

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql import SparkSession
-- MAGIC from pyspark.sql.functions import col, lag
-- MAGIC from pyspark.sql.window import Window
-- MAGIC from pyspark.ml.feature import VectorAssembler

-- COMMAND ----------

USE catalog hive_metastore;
USE lakehouse_in_action;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Create small dataset for testing

-- COMMAND ----------



CREATE OR REPLACE TABLE parkinsons_tiny_fe_n_metadata_w_time AS (SELECT *, row_number() OVER (PARTITION BY `Id` ORDER BY Init) TimeStep FROM parkinsons_tiny_fe_n_metadata)

-- COMMAND ----------

CREATE OR REPLACE TABLE hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids AS (SELECT * FROM (SELECT DISTINCT id FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog) TABLESAMPLE (5 ROWS));
SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col
-- MAGIC
-- MAGIC subs = sql("""SELECT Subject FROM hive_metastore.lakehouse_in_action.parkinsons_defog_metadata WHERE id IN (SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids)""")
-- MAGIC subs = subs.select(col("Subject")).rdd.flatMap(lambda x: x).collect()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC sql(
-- MAGIC     f"""CREATE OR REPLACE TABLE parkinsons_tiny_subjects AS (SELECT * FROM parkinsons_subjects WHERE Subject IN {tuple(subs)})"""
-- MAGIC )
-- MAGIC sql(
-- MAGIC     f"""CREATE OR REPLACE TABLE parkinsons_tiny_defog_metadata AS (SELECT * FROM parkinsons_defog_metadata WHERE Subject IN {tuple(subs)})"""
-- MAGIC )
-- MAGIC sql(
-- MAGIC     f"""CREATE OR REPLACE TABLE parkinsons_tiny_tdcsfog_metadata AS (SELECT * FROM parkinsons_tdcsfog_metadata WHERE Subject IN {tuple(subs)})"""
-- MAGIC )
-- MAGIC
-- MAGIC

-- COMMAND ----------

CREATE
OR REPLACE TABLE parkinsons_tiny_fe_n_metadata AS (
  SELECT
    fe.`Id`,
    fe.Init,
    fe.Completion,
    fe.Kinetic,
    fe.`Type`,
    dm.Subject,
    dm.Visit,
    null as Test,
    dm.Medication
  from
    parkinsons_fog_events fe
    JOIN parkinsons_tiny_defog_metadata dm ON fe.Id = dm.Id
  UNION
  SELECT
    fe.`Id`,
    fe.Init,
    fe.Completion,
    fe.Kinetic,
    fe.`Type`,
    tm.Subject,
    tm.Visit,
    tm.Test,
    tm.Medication
  from
    parkinsons_fog_events fe
    JOIN parkinsons_tiny_tdcsfog_metadata tm ON fe.Id = tm.Id
)

-- COMMAND ----------

CREATE
OR REPLACE TABLE hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids AS (
  SELECT
    DISTINCT id
  FROM
    hive_metastore.lakehouse_in_action.parkinsons_tiny_fe_n_metadata
);
SELECT
  *
FROM
  hive_metastore.lakehouse_in_action.parkinsons_tiny_train_defog_ids;

-- COMMAND ----------

CREATE
OR REPLACE TABLE parkinsons_tiny_tasks AS (
  SELECT
    *
  from
    parkinsons_defog_tasks
  WHERE
    `Id` IN (SELECT * FROM parkinsons_tiny_train_defog_ids)
)

-- COMMAND ----------

CREATE
OR REPLACE TABLE parkinsons_tiny_train AS (
  SELECT
    int(`Time`) as TimeStep,
    CASE
      WHEN (
        Task == true
        AND Valid == true
        AND (StartHesitation + Turn + Walking) > 0
      ) THEN 1
      ELSE 0
    END AS EventInProgress,
    *
  from
    parkinsons_train_defog
  WHERE
    `Id` IN (
      SELECT
        *
      FROM
        parkinsons_tiny_train_defog_ids
    )
)

-- COMMAND ----------

CREATE OR REPLACE TABLE parkinsons_unambig_defog_train AS (
with start_time as (select `id`,  min(`Time`) as StartTime FROM lakehouse_in_action.parkinsons_train_defog WHERE (Task == true AND Valid == true AND (StartHesitation + Turn + Walking) > 0) GROUP BY `id`),
end_time as (select  `id`,  max(`Time`) as EndTime FROM lakehouse_in_action.parkinsons_train_defog WHERE (Task == true AND Valid == true AND (StartHesitation + Turn + Walking) > 0) GROUP BY `id`)
SELECT 
int(`Time`) as TimeStep,
tdf.*,
CASE 
WHEN (Task == true AND Valid == true AND (StartHesitation + Turn + Walking) > 0)
THEN 1 ELSE 0 END AS EventInProgress,
isnotnull(st.StartTime) as EventStart,
isnotnull(et.EndTime) as EventEnd
FROM lakehouse_in_action.parkinsons_train_defog tdf
left join start_time st ON tdf.id = st.id AND tdf.`Time` = st.StartTime
left join end_time et ON tdf.id = et.id AND tdf.`Time` = et.EndTime
order by `id`, int(`Time`))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Lag variables

-- COMMAND ----------

SELECT * FROM parkinsons_unambig_defog_train LIMIT 5

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC my_window = Window.partitionBy(col("id")).orderBy(col("TimeStep"))
-- MAGIC lag_columns = ["AccV", "AccML", "AccAP"]

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC df = sql("SELECT * FROM parkinsons_tiny_train")
-- MAGIC
-- MAGIC # Calculate lagged features
-- MAGIC lag_window = 1  # Adjust the window size as needed
-- MAGIC
-- MAGIC df = df.select(
-- MAGIC     "*",
-- MAGIC     *[lag(c, offset=lag_window).over(my_window).alias(f"{c}_lag_{lag_window}") for c in lag_columns
-- MAGIC     ]
-- MAGIC )
-- MAGIC display(df.groupby("id").count())

-- COMMAND ----------


# Drop rows with null values resulting from lagged features
data = data.dropna()

# Add lagged values as features
lags = [1, 2, 3]  # Specify the lag values you want to use
for lag_value in lags:
    data = data.withColumn("lag_" + str(lag_value), lag(col("value"), lag_value).over(Window.orderBy("timestamp")))

# Add rolling window statistics as features
window_sizes = [3, 5]  # Specify the window sizes you want to use
for window_size in window_sizes:
    data = data.withColumn("rolling_mean_" + str(window_size), col("value").rolling(window_size).mean().over(Window.orderBy("timestamp")))
    data = data.withColumn("rolling_std_" + str(window_size), col("value").rolling(window_size).stddev().over(Window.orderBy("timestamp")))



-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Testing feature importance
-- MAGIC Most of this is copied from an AutoML notebook

-- COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "EventInProgress"

-- COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector

supported_cols = ["AccV", "AccAP", "AccML"]
col_selector = ColumnSelector(supported_cols)

-- COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), supported_col))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["AccV", "AccAP", "AccML"])]

-- COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

-- COMMAND ----------

# AutoML completed train - validation - test split internally and used _automl_split_col_df25 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_df25 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_df25 == "val"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_df25 == "test"]

# Separate target column from features and drop _automl_split_col_df25
X_train = split_train_df.drop([target_col, "_automl_split_col_df25"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_df25"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_df25"], axis=1)
y_test = split_test_df[target_col]

-- COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

def objective(params):
  with mlflow.start_run(experiment_id="3341193441114056") as mlflow_run:
    sklr_classifier = LogisticRegression(**params)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("classifier", sklr_classifier),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True)

    model.fit(X_train, y_train)

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(target_col):y_train}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": True }
    )
    sklr_training_metrics = training_eval_result.metrics
    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(target_col):y_val}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "val_" , "pos_label": True }
    )
    sklr_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(target_col):y_test}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": True }
    )
    sklr_test_metrics = test_eval_result.metrics

    loss = sklr_val_metrics["val_f1_score"]

    # Truncate metric key names so they can be displayed together
    sklr_val_metrics = {k.replace("val_", ""): v for k, v in sklr_val_metrics.items()}
    sklr_test_metrics = {k.replace("test_", ""): v for k, v in sklr_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": sklr_val_metrics,
      "test_metrics": sklr_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

-- COMMAND ----------

space = {
  "C": 0.001924334647693606,
  "penalty": "l2",
  "random_state": 171701030,
}

-- COMMAND ----------

trials = Trials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=1,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

-- COMMAND ----------

if shap_enabled:
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=171701030)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=171701030)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False, nsamples=500)
    summary_plot(shap_values, example, class_names=model.classes_)
