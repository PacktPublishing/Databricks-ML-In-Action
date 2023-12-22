# Databricks notebook source
# MAGIC %md 
# MAGIC Chapter 3 

# COMMAND ----------



# COMMAND ----------

dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "")


# COMMAND ----------

import os 

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

MAIN_DIR_UC = f"/Volumes/{catalog_name}/{schema_name}/intel_image_clf/raw_images"
MAIN_DIR2Write = "/Volumes/{catalog_name}/{schema_name}/intel_image_clf/"
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
            # As of now DeltaTorch Loader reuqires you to have tables in delta format under Volumes when UC is used
            # in the future the functionality will involve. 
            df.write.format("delta").mode("append").save("{path2write}{name2write}")
        if returnDF:
            return df

# COMMAND ----------

prep_data2delta(
    train_dir,
    outcomes,
    "train_imgs_main.delta",
    write2detla=True,
    path2write=MAIN_DIR2Write
    returnDF=None,
)

# COMMAND ----------

prep_data2delta(
    valid_dir,
    outcomes,
    "valid_imgs_main.delta",
    path2write=MAIN_DIR2Write
    write2detla=True,
    returnDF=None,
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Adding a few addiitonal options

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE delta.`/Volumes/YOURCATALOG/YOURSCHEMA/intel_image_clf/valid_imgs_main.delta`

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE delta.`/Volumes/YOURCATALOG/YOURSCHEMA/intel_image_clf/train_imgs_main.delta`

# COMMAND ----------

# MAGIC %sql 
# MAGIC ALTER TABLE delta.`/Volumes/YOURCATALOG/YOURSCHEMA/intel_image_clf/train_imgs_main.delta` SET TBLPROPERTIES ('delta.enableDeletionVectors' = false);
# MAGIC ALTER TABLE delta.`/Volumes/YOURCATALOG/YOURSCHEMA/intel_image_clf/valid_imgs_main.delta` SET TBLPROPERTIES ('delta.enableDeletionVectors' = false);

# COMMAND ----------


