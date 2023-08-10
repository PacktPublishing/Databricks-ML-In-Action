# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Feature Engineering for Event Detection
# MAGIC Your objective is to detect the start and stop of each freezing episode and the occurrence in these series of three types of freezing of gait events: Start Hesitation, Turn, and Walking.

# COMMAND ----------

# MAGIC %run ./../../global-setup $project_name=parkinsons-freezing_gait_prediction $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM silver_subjects

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE defog_enriched_tasks AS (
# MAGIC   SELECT
# MAGIC     dm.Id as TaskID,
# MAGIC     tk.Begin,
# MAGIC     tk.`End`,
# MAGIC     dm.Subject,
# MAGIC     dm.Medication,
# MAGIC     dm.Visit,
# MAGIC     ss.Age,
# MAGIC     ss.Sex,
# MAGIC     ss.YearsSinceDx,
# MAGIC     ss.UPDRSIII_On,
# MAGIC     ss.UPDRSIII_Off,
# MAGIC     ss.NFOGQ
# MAGIC   FROM
# MAGIC     parkinsons_defog_metadata dm
# MAGIC     LEFT JOIN silver_subjects ss ON dm.Subject = ss.Subject
# MAGIC     AND dm.Visit = ss.Visit
# MAGIC     LEFT JOIN parkinsons_defog_tasks tk ON dm.Id = tk.Id
# MAGIC )

# COMMAND ----------

# DBTITLE 1,Create a training table for building features
# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE defog_training AS (
# MAGIC   SELECT
# MAGIC     tr.*,
# MAGIC     tr.`id` as TaskID,
# MAGIC     det.Subject,
# MAGIC     det.Medication,
# MAGIC     det.Visit,
# MAGIC     det.Age,
# MAGIC     det.Sex,
# MAGIC     det.YearsSinceDx,
# MAGIC     det.UPDRSIII_On,
# MAGIC     det.UPDRSIII_Off,
# MAGIC     det.NFOGQ
# MAGIC   FROM
# MAGIC     parkinsons_train_defog tr
# MAGIC     LEFT JOIN defog_enriched_tasks det ON tr.id = det.TaskID
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Rolling AVG variables

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE defog_temp AS (
# MAGIC   SELECT
# MAGIC     *
# MAGIC   FROM
# MAGIC     defog_training
# MAGIC   LIMIT
# MAGIC     1000
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM defog_temp

# COMMAND ----------

from pyspark.sql.functions import avg
from pyspark.sql.window import Window

window_size = "1 day"  # set the window size as per your requirements

df = spark.table("defog_temp") # assuming the table 'defog_temp' is registered in Spark

# create a window specification
window_spec = Window.orderBy("Time").rangeBetween(-(60*60*24), 0)

# calculate the rolling average using avg() function over the created window
df_rolling_avg = df.select(
        "date_time",
        "AccV",
        avg("AccV").over(window_spec).alias("rolling_avg_AccV")
    )
df_rolling_avg.show()

# COMMAND ----------

from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window

my_window = Window.partitionBy(col("id")).orderBy(col("TimeStep"))
lag_columns = ["AccV", "AccML", "AccAP"]

# COMMAND ----------

df = sql("""SELECT * FROM parkinsons_updated_defog_train""")

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


