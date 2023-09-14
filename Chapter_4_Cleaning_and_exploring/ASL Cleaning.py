# Databricks notebook source
# MAGIC %md
# MAGIC # ASL Fingerspelling Cleaning the Dataset
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
# MAGIC
# MAGIC We need to reduce the sequeces down to those with enough non-null data points.
# MAGIC
# MAGIC ##Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess the data
# MAGIC
# MAGIC Again, the competition provides code specific to this dataset.
# MAGIC
# MAGIC Reference:
# MAGIC
# MAGIC https://www.kaggle.com/code/gusthema/asl-fingerspelling-recognition-w-tensorflow/notebook
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create x,y,z label names from coordinates

# COMMAND ----------

# Pose coordinates for hand movement.
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

# COMMAND ----------

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

# COMMAND ----------

# MAGIC %md
# MAGIC Create feature column names from the extracted coordinates.

# COMMAND ----------

FEATURE_COLUMNS = X + Y + Z

# COMMAND ----------

# MAGIC %md
# MAGIC Store ids of each coordinate labels to lists

# COMMAND ----------

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Spark and PySpark Pandas to reduce the sequences used for training
# MAGIC
# MAGIC Using the extracted landmarks and phrases to limit to those with more non-null values than 2x the length of the phrase

# COMMAND ----------

from pyspark.sql.functions import greatest, col, length
import pyspark.pandas as ps

# COMMAND ----------

df = spark.table("train_landmarks").select("sequence_id", *FEATURE_COLUMNS)
df = df.withColumn(
    "num_rh_null",
    sum(df[colm].isNull().cast("int") for colm in RHAND_IDX),
).withColumn(
    "num_lh_null",
    sum(df[colm].isNull().cast("int") for colm in LHAND_IDX),
)

# COMMAND ----------

rhdf = df.filter(col('num_rh_null')==0).groupBy('sequence_id').count().withColumnRenamed("count","rh_nn_rows")
lhdf = df.filter(col('num_lh_null')==0).groupBy('sequence_id').count().withColumnRenamed("count","lh_nn_rows")

mdf = spark.table("training_metadata").withColumn("phrase_length", length(col("phrase")))
mdf = mdf.join(lhdf, on='sequence_id', how='left').join(rhdf, on='sequence_id', how='left').fillna({'lh_nn_rows': 0,'rh_nn_rows': 0})
mdf = mdf.withColumn('max_nn_rows', greatest(col("lh_nn_rows"), col("rh_nn_rows")))

# COMMAND ----------

mdf.filter(2*col('phrase_length')<col('max_nn_rows')).write.mode('overwrite').saveAsTable("cleaned_training_metadata")

# COMMAND ----------

# MAGIC %md
# MAGIC ##below this, I want to keep for my explaination

# COMMAND ----------

import numpy as np

seq_id = '1983865658'
phrase = 'jeramy duran'
tdf= spark.table("train_landmarks").select("sequence_id", *FEATURE_COLUMNS).filter(col('sequence_id')==seq_id)
pdf = tdf.toPandas()

parquet_numpy = tdf.pandas_api().to_numpy()

frames = parquet_numpy

# Calculate the number of NaN values in each hand landmark
r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
no_nan = max(r_nonan, l_nonan)

if 2 * len(phrase) < no_nan:
  print(no_nan)

# COMMAND ----------

display(np.isnan(frames[:, RHAND_IDX]))

# COMMAND ----------

display(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1))

# COMMAND ----------

len(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1))

# COMMAND ----------

np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)

# COMMAND ----------


