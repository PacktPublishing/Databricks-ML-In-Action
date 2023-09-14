# Databricks notebook source
# MAGIC %md
# MAGIC # ASL Fingerspelling
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
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
# MAGIC For convenience and efficiency, we will rearrange the data so that each parquet file contains the landmark data along with the phrase it represents. This way we don't have to switch between train.csv and its parquet file. 
# MAGIC
# MAGIC We will save the new data in the [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format. `TFRecord` format is a simple format for storing a sequence of binary records. Storing and loading the data using `TFRecord` is much more efficient and faster.
# MAGIC
# MAGIC Reference:
# MAGIC
# MAGIC https://www.kaggle.com/code/gusthema/asl-fingerspelling-recognition-w-tensorflow/notebook
# MAGIC
# MAGIC https://www.kaggle.com/code/irohith/aslfr-preprocess-dataset
# MAGIC
# MAGIC https://www.kaggle.com/code/shlomoron/aslfr-parquets-to-tfrecords-cleaned

# COMMAND ----------

# Pose coordinates for hand movement.
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = (
    [f"x_right_hand_{i}" for i in range(21)]
    + [f"x_left_hand_{i}" for i in range(21)]
    + [f"x_pose_{i}" for i in POSE]
)
Y = (
    [f"y_right_hand_{i}" for i in range(21)]
    + [f"y_left_hand_{i}" for i in range(21)]
    + [f"y_pose_{i}" for i in POSE]
)
Z = (
    [f"z_right_hand_{i}" for i in range(21)]
    + [f"z_left_hand_{i}" for i in range(21)]
    + [f"z_pose_{i}" for i in POSE]
)

FEATURE_COLUMNS = X + Y + Z

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]

RHAND_COLS = [col for col in FEATURE_COLUMNS if "right" in col]
LHAND_COLS = [col for col in FEATURE_COLUMNS if "left" in col]
RPOSE_COLS = [col for col in FEATURE_COLUMNS if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_COLS = [col for col in FEATURE_COLUMNS if "pose" in col and int(col[-2:]) in LPOSE]


DOMINANT_COLS = (
    [f"x_hand_{i}" for i in range(21)]
    + [f"y_hand_{i}" for i in range(21)]
    + [f"z_hand_{i}" for i in range(21)]
    + [f"x_pose_{i}" for i in range(5)]
    + [f"y_pose_{i}" for i in range(5)]
    + [f"z_pose_{i}" for i in range(5)]
)

X_HAND_COLS = [col for col in DOMINANT_COLS if "x_hand" in col]
X_POSE_COLS = [col for col in DOMINANT_COLS if "x_pose" in col]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocess and write the dataset as TFRecords
# MAGIC
# MAGIC Using the extracted landmarks and phrases let us create new dataset and use petastorm to convert to tf format.

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql.functions import greatest, col, length, expr, when
import pyspark.pandas as ps
import tensorflow as tf

# COMMAND ----------

cmdf = spark.table("cleaned_training_metadata").select(
    "file_id","sequence_id", "phrase", "participant_id", "lh_nn_rows", "rh_nn_rows"
)
new_column = (
    when(col("lh_nn_rows") > col("rh_nn_rows"), "left")
    .when(col("lh_nn_rows") < col("rh_nn_rows"), "right")
    .when(col("lh_nn_rows") == col("rh_nn_rows"), "left")
    .otherwise("broken")
)
cmdf = cmdf.withColumn("dominant_hand", new_column)

display(cmdf)

# COMMAND ----------

display(cmdf.select("dominant_hand").distinct())

# COMMAND ----------

df = spark.table("train_landmarks").select("sequence_id", *FEATURE_COLUMNS)
df = df.join(cmdf, on="sequence_id")
display(df)

# COMMAND ----------

rh = df.select("sequence_id","phrase",*RHAND_COLS,*RPOSE_COLS).where(col("dominant_hand") == "right").toDF("sequence_id","phrase",*DOMINANT_COLS)
lh = df.select("sequence_id","phrase",*LHAND_COLS,*LPOSE_COLS).where(col("dominant_hand") == "left").toDF("sequence_id","phrase",*DOMINANT_COLS)

X_COLS = X_HAND_COLS + X_POSE_COLS
for lh_x_col in X_COLS:
  lh = lh.withColumn(lh_x_col,when(col(lh_x_col)!=0,1- col(lh_x_col)).otherwise(0))

featuresDF = rh.unionAll(lh)

# COMMAND ----------

# DBTITLE 1,https://docs.databricks.com/en/data-governance/unity-catalog/create-volumes.html
# MAGIC %sql
# MAGIC CREATE VOLUME lakehouse_in_action.asl_fingerspelling.asl_volume

# COMMAND ----------

# DBTITLE 1,https://docs.databricks.com/en/_extras/notebooks/source/unity-catalog-volumes.html
# MAGIC %sql
# MAGIC GRANT WRITE VOLUME
# MAGIC ON VOLUME lakehouse_in_action.asl_fingerspelling.asl_volume
# MAGIC TO `hayley.horn@databricks.com`;

# COMMAND ----------

dbutils.fs.mkdirs("/Volumes/lakehouse_in_action/asl_fingerspelling/asl_volume/tfrecords")


# COMMAND ----------

featuresDF.write.mode("overwrite").saveAsTable("feature_table")

# COMMAND ----------

# DBTITLE 1,I commented this out so it doesn't accidentally get run again
# stem = "/Volumes/lakehouse_in_action/asl_fingerspelling/asl_volume/tfrecords/"

# features_pd = featuresDF.toPandas()
# features_numpy = features_pd.to_numpy()

# files = cmdf.select('file_id').distinct().toLocalIterator()

# for row in files:
#   file_id = row.file_id
#   tf_file = f"{stem}{file_id}.tfrecord"
#   sequences = cmdf.filter(col("file_id") == file_id).select('sequence_id','phrase').toPandas()
#   with tf.io.TFRecordWriter(tf_file) as file_writer:
#     for seq_id,phrase in zip(sequences.sequence_id,sequences.phrase):
#         frames = features_numpy[features_pd.index == seq_id]
#         features = {DOMINANT_COLS[i]: tf.train.Feature(
#                       float_list=tf.train.FloatList(value=frames[:, i+2])) for i in range(len(DOMINANT_COLS))}
#         features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
#         record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
#         file_writer.write(record_bytes)

# COMMAND ----------


