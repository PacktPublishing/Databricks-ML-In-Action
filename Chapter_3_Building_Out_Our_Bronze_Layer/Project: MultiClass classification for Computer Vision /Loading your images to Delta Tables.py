# Databricks notebook source
# MAGIC %md 
# MAGIC Chapter 3 
# MAGIC
# MAGIC ## Intel Mulilable Image Classification - Ingest your data into Delta 
# MAGIC

# COMMAND ----------

dbutils.widgets.dropdown(name='Reset', defaultValue='True', choices=['True', 'False'], label="Reset: Drop previous table")

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

import os 
# be sure you have executed your code in Ch2! 
MAIN_DIR_UC = f"/Volumes/{catalog}/{database_name}/intel_image_clf/raw_images"
MAIN_DIR2Write = "/Volumes/{catalog}/{database_name}/intel_image_clf/"
data_dir_Train = f"{MAIN_DIR_UC}/seg_train"
data_dir_Test = f"{MAIN_DIR_UC}/seg_test"
data_dir_pred = f"{MAIN_DIR_UC}/seg_pred/seg_pred"

train_dir = data_dir_Train + "/seg_train"
valid_dir = data_dir_Test + "/seg_test"
pred_files = [os.path.join(data_dir_pred, f) for f in os.listdir(data_dir_pred)]

labels_dict_train = {f"{f}":len(os.listdir(os.path.join(train_dir, f))) for f in os.listdir(train_dir)}
labels_dict_valid = {f"{f}":len(os.listdir(os.path.join(valid_dir, f))) for f in os.listdir(valid_dir)}

outcomes = os.listdir(train_dir)
print(outcomes)

# COMMAND ----------

delta_train_name = "train_imgs_main.delta"
delta_val_name = "valid_imgs_main.delta"

if bool(dbutils.widgets.get('Reset')):
  dbutils.fs.rm(f"{MAIN_DIR2Write}{delta_train_name}")
  dbutils.fs.rm(f"{MAIN_DIR2Write}{delta_val_name}")

# COMMAND ----------

from pyspark.sql import functions as f


def prep_data2delta(
    dir_name,
    outcomes,
    name2write,
    path2write="YOUR_PATH",
    write2detla=True,
    returnDF=None,
):

    mapping_dict = {
        "buildings": 0,
        "sea": 1,
        "glacier": 2,
        "forest": 3,
        "street": 4,
        "mountain": 5,
    }
    # As we have multi label problem we will loop over labels to save them all under 1 main training set 
    for LABEL_NAME in outcomes:
        df = (
            spark.read.format("binaryfile")
            .option("recursiveFileLookup", "true")
            .load(f"{dir_name}/{LABEL_NAME}")
            .withColumn("label_name", f.lit(f"{LABEL_NAME}"))
            .withColumn("label_id", f.lit(f"{mapping_dict[LABEL_NAME]}").astype("int"))
            .withColumn("image_name", f.split(f.col("path"), "/").getItem(9))
            .withColumn(
                "id", f.split(f.col("image_name"), ".jpg").getItem(0).astype("int")
            )
        )
        if write2delta:
            df.write.format("delta").mode("append").save("{path2write}{name2write}")
        if returnDF:
            return df

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Note 
# MAGIC
# MAGIC  As of now DeltaTorch Loader requires you to have tables in delta format under Volumes when UC is used in the future the functionality will involve. It's simply due to some security permissions on the read/write that UC has. <br>
# MAGIC  If you are not using UC you should be able to read delta tables directly from the managed tables. 
# MAGIC
# MAGIC  You could also simply keep filenames under your Delta Table instead of images itself and collect them to pass to the main trainer. Why keeping anything under Delta is important, simply to avoid having duplicates and to control the list that have been used during the training as you can pass the Delta version to the MLFlow during the tracking. 
# MAGIC
# MAGIC  There is also an option to read your data using Petastorm library, but today it's not recommended as it requires you to carefully read the doc to understand pitfals it has (mainly memory usage due to the dat caching and the fact it's using parquet files and not Delta, so you are consuming all versions of your files). 

# COMMAND ----------

prep_data2delta(
    train_dir,
    outcomes,
    delta_train_name,
    write2detla=True,
    path2write=MAIN_DIR2Write
    returnDF=None,
)

# COMMAND ----------

prep_data2delta(
    valid_dir,
    outcomes,
    delta_val_name,
    path2write=MAIN_DIR2Write
    write2detla=True,
    returnDF=None,
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Adding a few addiitonal options

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE delta.`/Volumes/$catalog/$database_name/intel_image_clf/valid_imgs_main.delta`

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE delta.`/Volumes/$catalog/$database_name/intel_image_clf/train_imgs_main.delta`

# COMMAND ----------

# MAGIC %sql 
# MAGIC ALTER TABLE delta.`/Volumes/$catalog/$database_name/intel_image_clf/train_imgs_main.delta` SET TBLPROPERTIES ('delta.enableDeletionVectors' = false);
# MAGIC ALTER TABLE delta.`/Volumes/$catalog/$database_name/intel_image_clf/valid_imgs_main.delta` SET TBLPROPERTIES ('delta.enableDeletionVectors' = false);

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM delta.`/Volumes/$catalog/$schema/intel_image_clf/valid_imgs_main.delta`

# COMMAND ----------


