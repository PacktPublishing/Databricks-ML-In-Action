# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Feature Engineering for Event Detection
# MAGIC Your objective is to detect the start and stop of each freezing episode and the occurrence in these series of three types of freezing of gait events: Start Hesitation, Turn, and Walking.

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=parkinsons-freezing_gait_prediction

# COMMAND ----------

# MAGIC %sql
# MAGIC USE catalog hive_metastore;
# MAGIC USE lakehouse_in_action;

# COMMAND ----------

# DBTITLE 1,Create a table for building features from
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE hive_metastore.lakehouse_in_action.parkinsons_updated_defog_train AS (
# MAGIC with start_time as (select `id`,  min(`Time`) as StartTime FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog WHERE ((StartHesitation + Turn + Walking) > 0) GROUP BY `id`),
# MAGIC end_time as (select  `id`,  max(`Time`) as EndTime FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog WHERE ((StartHesitation + Turn + Walking) > 0) GROUP BY `id`)
# MAGIC SELECT 
# MAGIC int(`Time`) as TimeStep,
# MAGIC float(tdf.AccV) as AccV,
# MAGIC float(tdf.AccML) as AccML,
# MAGIC float(tdf.AccAP) as AccAP,
# MAGIC tdf.`id`,
# MAGIC CASE 
# MAGIC WHEN StartHesitation == 1 THEN "StartHesitation"
# MAGIC WHEN Turn == 1 THEN "Turn"
# MAGIC WHEN Walking == 1 THEN "Walking"
# MAGIC ELSE "None" END AS StringLabel,
# MAGIC CASE 
# MAGIC WHEN ((StartHesitation + Turn + Walking) > 0)
# MAGIC THEN 1 ELSE 0 END AS EventInProgress,
# MAGIC isnotnull(st.StartTime) as EventStart,
# MAGIC isnotnull(et.EndTime) as EventEnd
# MAGIC FROM hive_metastore.lakehouse_in_action.parkinsons_train_defog tdf
# MAGIC left join start_time st ON tdf.id = st.id AND tdf.`Time` = st.StartTime
# MAGIC left join end_time et ON tdf.id = et.id AND tdf.`Time` = et.EndTime
# MAGIC order by `id`, int(`Time`))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Lag variables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM parkinsons_updated_defog_train LIMIT 10

# COMMAND ----------

from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window

my_window = Window.partitionBy(col("id")).orderBy(col("TimeStep"))
lag_columns = ["AccV", "AccML", "AccAP"]

# COMMAND ----------

df = sql("""SELECT * FROM hive_metastore.lakehouse_in_action.parkinsons_updated_defog_train""")

lag_window = 1  # Adjust the window size as needed

# Add lagged values as features
data = df.select(
    "*",
    *[lag(c, offset=lag_window).over(my_window).alias(f"{c}_lag_{lag_window}") for c in lag_columns
    ]
)

new_columns = [f"{c}_lag_{lag_window}" for c in lag_columns]
# Drop rows with null values resulting from lagged features
data = data.dropna()

# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a model to evaluate features against

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.sql.functions import *
import mlflow

label_col = "StringLabel"
feature_cols = lag_columns + new_columns

mlflow.autolog()

# COMMAND ----------

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol=label_col, outputCol="IndexedLabel").fit(data)

# Assemble numerical values into a single feature vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="Features")

# Split the data into train and test sets with stratification
train_fraction = 0.7  # Adjust the train fraction as needed
splits = data.select(label_col).distinct().rdd.flatMap(lambda x: x).collect()
stratified_split = data.stat.sampleBy(label_col, fractions={label: train_fraction for label in splits}, seed=416)

# Split the data into training and test sets (30% held out for testing)
trainingData, testData = stratified_split.randomSplit([train_fraction, 1 - train_fraction], seed=416)


# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="IndexedLabel", featuresCol="Features")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)
# Make predictions.
predictions = model.transform(testData)

# COMMAND ----------


# Select example rows to display.
predictions.select("prediction", "IndexedLabel", "Features").show(5)

# Evaluate the classifier using various metrics
evaluator = MulticlassClassificationEvaluator(labelCol="IndexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

evaluator = MulticlassClassificationEvaluator(labelCol="IndexedLabel", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(predictions)
print("Precision:", precision)

evaluator = MulticlassClassificationEvaluator(labelCol="IndexedLabel", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(predictions)
print("Recall:", recall)

evaluator = MulticlassClassificationEvaluator(labelCol="IndexedLabel", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print("F1-score:", f1_score)

# COMMAND ----------

from shap import KernelExplainer, summary_plot
import pandas as pd

# Convert the PySpark model to a Python model
model_py = model.stages[-1]

# Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
train_sample = assembler.transform(trainingData).select("Features").toPandas().sample(n=100, random_state=171701030)

# Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
example = assembler.transform(testData).select("Features").toPandas().sample(n=100, random_state=171701030)

# Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
predict = lambda x: model_py.predict(pd.DataFrame(x, columns="Features"))
explainer = KernelExplainer(predict, train_sample, link="identity")
shap_values = explainer.shap_values(example, l1_reg=False, nsamples=500)
summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

from shap import KernelExplainer, summary_plot

# Extract the feature vector from the test data
featureData = assembler.transform(testData).select("Features")

# Convert the feature data DataFrame to a Pandas DataFrame
featureData_pd = featureData.toPandas()

# Convert the PySpark model to a Python model
model_py = model.stages[-1]

# Initialize the SHAP explainer with the PySpark model
explainer = KernelExplainer(model_py)

# Calculate SHAP values for the test data
shap_values = explainer.shap_values(featureData_pd)

# Calculate the feature importance based on SHAP values
feature_importance = shap_values.abs.mean(axis=0)

# Print the feature importance
print("Feature Importance:")
for feature, importance in zip(featureData.columns, feature_importance):
    print(feature, ":", importance)

summary_plot(shap_values, featureData_pd, class_names=model.classes_)

# COMMAND ----------

def calculate_shap(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for X in iterator:
        yield pd.DataFrame(
            explainer.shap_values(np.array(X), check_additivity=False)[0],
            columns=columns_for_shap_calculation,
        )

return_schema = StructType()
for feature in columns_for_shap_calculation:
    return_schema = return_schema.add(StructField(feature, FloatType()))

shap_values = df.mapInPandas(calculate_shap, schema=return_schema)

# COMMAND ----------

display(predictions.groupBy(['IndexedLabel','prediction']).count())

# COMMAND ----------


