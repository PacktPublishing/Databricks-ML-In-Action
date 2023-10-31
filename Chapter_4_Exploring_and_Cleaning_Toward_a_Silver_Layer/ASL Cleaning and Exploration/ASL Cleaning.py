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

# MAGIC %run ../../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

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

display(mdf.filter(2*col('phrase_length')<col('max_nn_rows')).take(10))

# COMMAND ----------


