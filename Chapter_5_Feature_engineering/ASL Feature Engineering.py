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
# MAGIC Again, the competition provides code specific to this dataset. However, in the case of batch inference, distributed feature engineering it is ideal. We want to distribute the work across workers rather than a single node. We demonstrate a way to do this here. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine the dominant hand
# MAGIC
# MAGIC Using the information found in the cleaning process, determine the dominant hand in each sequence.

# COMMAND ----------

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
    .when(col("lh_nn_rows") == col("rh_nn_rows"), "same")
    .otherwise("broken")
)
cmdf = cmdf.withColumn("dominant_hand", new_column)

display(cmdf)

# COMMAND ----------

display(cmdf.select("dominant_hand").groupBy("dominant_hand").count())

# COMMAND ----------

df = spark.table("train_landmarks").select("sequence_id", *FEATURE_COLUMNS)
df = df.join(cmdf, on="sequence_id")
display(df.filter(col("dominant_hand")=="same"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select only the dominant hand coordinates
# MAGIC
# MAGIC When there isn't adominant hand choose the right hand to be dominant

# COMMAND ----------

rh = df.select("sequence_id","phrase",*RHAND_COLS,*RPOSE_COLS).where(col("dominant_hand") == "right").toDF("sequence_id","phrase",*DOMINANT_COLS)
sh = df.select("sequence_id","phrase",*RHAND_COLS,*RPOSE_COLS).where(col("dominant_hand") == "same").toDF("sequence_id","phrase",*DOMINANT_COLS)
rh = rh.unionAll(sh)
lh = df.select("sequence_id","phrase",*LHAND_COLS,*LPOSE_COLS).where(col("dominant_hand") == "left").toDF("sequence_id","phrase",*DOMINANT_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reflect the left hand to align as a right hand
# MAGIC For the left hand dataframe, flip the x coordinates horizontally. The function `y = 1 - x` reflects the function `y = x` horizontally at `y = 1/2` for values of x between 0 and 1

# COMMAND ----------

X_COLS = X_HAND_COLS + X_POSE_COLS
for lh_x_col in X_COLS:
  lh = lh.withColumn(lh_x_col,when(col(lh_x_col)!=0,1 - col(lh_x_col)).otherwise(0))

featuresDF = rh.unionAll(lh)

# COMMAND ----------

featuresDF.write.mode("overwrite").saveAsTable("feature_table")

# COMMAND ----------

# MAGIC %md First instantiate a feature store client

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md Use create_feature_table API to create the feature store tables and define the schema and unique ID keys. We use the features_df argument to write the data to Feature Store.

# COMMAND ----------

fs.create_feature_table(
    name="lakehouse_in_action.asl_fingerspelling.ASL_training_fs_table",
    keys=["sequence_id"],
    features_df=featuresDF,
    description="ASL fingerspelling feature store table for training",
)
