# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ## ASL Fingerspelling - ASL Feature Engineering
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)

# COMMAND ----------

# MAGIC %md ## Run Setup

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

from pyspark.sql.functions import greatest, col, length, expr, when, abs
from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.pandas as ps

# COMMAND ----------

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
# MAGIC For the left hand dataframe, flip the x coordinates horizontally. The function `y = |1 - x|` reflects the function `y = x` horizontally at `x = 1/2` for values of x between 0 and 1

# COMMAND ----------

X_COLS = X_HAND_COLS + X_POSE_COLS
for lh_x_col in X_COLS:
  lh = lh.withColumn(lh_x_col,when(col(lh_x_col)!=0,abs(1 - col(lh_x_col))).otherwise(0))

featuresDF = rh.unionAll(lh)

# COMMAND ----------

featuresDF.write.mode("overwrite").saveAsTable("feature_table")

# COMMAND ----------

# MAGIC %md First instantiate a feature engineering client

# COMMAND ----------

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md Use create_table API to create the feature tables and define the schema and unique ID keys. We use the features_df to specify the dataframe.

# COMMAND ----------

fe.create_table(
    name="lakehouse_in_action.asl_fingerspelling.ASL_training_table",
    primary_keys=["sequence_id"],
    features_df=featuresDF,
    description="ASL fingerspelling feature table for training",
)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id, rank

# Create and index to maintain current order
featuresDF_indexed = featuresDF.select("*").withColumn("step", monotonically_increasing_id())
# Define window
window = Window.partitionBy(featuresDF_indexed['sequence_id']).orderBy(featuresDF_indexed['step'])
# Create column
featuresDF_indexed = featuresDF_indexed.select('*', rank().over(window).alias('sequence_step'))
display(featuresDF_indexed.take(50))

# COMMAND ----------

fe.create_table(
    name="lakehouse_in_action.asl_fingerspelling.ASL_training_table",
    primary_keys=["sequence_id","sequence_step"],
    df=featuresDF_indexed,
    description="ASL fingerspelling feature table for training",
)
