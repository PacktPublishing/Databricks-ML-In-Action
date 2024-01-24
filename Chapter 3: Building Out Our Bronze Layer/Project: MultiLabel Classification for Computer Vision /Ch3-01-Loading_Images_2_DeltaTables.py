# Databricks notebook source
# MAGIC %md 
# MAGIC Chapter 3 
# MAGIC
# MAGIC ## Intel Mulilable Image Classification - Ingest your data into Delta 
# MAGIC

# COMMAND ----------

dbutils.widgets.dropdown(name='Reset', defaultValue='True', choices=['True', 'False'], label="Reset: Delete previous data")

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

delta_train_name = "train_imgs_main.delta"
delta_val_name = "valid_imgs_main.delta"

if bool(dbutils.widgets.get('Reset')):
  !rm -rf {main_dir_2write}{delta_train_name}
  !rm -rf {main_dir_2write}{delta_val_name}

# COMMAND ----------

from pyspark.sql import functions as f


def prep_data2delta(
    dir_name,
    outcomes,
    name2write,
    path2write="YOUR_PATH",
    write2delta=True,
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
    for label_name in outcomes:
        df = (
            spark.read.format("binaryfile")
            .option("recursiveFileLookup", "true")
            .load(f"{dir_name}/{label_name}")
            .withColumn("label_name", f.lit(f"{label_name}"))
            .withColumn("label_id", f.lit(f"{mapping_dict[label_name]}").astype("int"))
            .withColumn("image_name", f.split(f.col("path"), "/").getItem(10))
            .withColumn(
                "id", f.split(f.col("image_name"), ".jpg").getItem(0).astype("int")
            )
        )
        if write2delta:
            df.write.format("delta").mode("append").save(f"{path2write}{name2write}")
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
    write2delta=True,
    path2write=main_dir_2write,
    returnDF=None,
)

# COMMAND ----------

prep_data2delta(
    valid_dir,
    outcomes,
    delta_val_name,
    path2write=main_dir_2write,
    write2delta=True,
    returnDF=None,
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Adding a few addiitonal options

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- you can set up a widget to a notebook and consume widgets via $ variables with SQL
# MAGIC --OPTIMIZE delta.`/Volumes/$catalog/$database_name/files/intel_image_clf/valid_imgs_main.delta`
# MAGIC OPTIMIZE delta.`/Volumes/ml_in_action/cv_clf/files/intel_image_clf/train_imgs_main.delta`

# COMMAND ----------

# MAGIC %sql 
# MAGIC OPTIMIZE delta.`/Volumes/ml_in_action/cv_clf/files/intel_image_clf/valid_imgs_main.delta`

# COMMAND ----------

# MAGIC %sql 
# MAGIC ALTER TABLE delta.`/Volumes/ml_in_action/cv_clf/files/intel_image_clf/train_imgs_main.delta` SET TBLPROPERTIES ('delta.enableDeletionVectors' = false);
# MAGIC ALTER TABLE delta.`/Volumes/ml_in_action/cv_clf/files/intel_image_clf/valid_imgs_main.delta` SET TBLPROPERTIES ('delta.enableDeletionVectors' = false);

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM delta.`/Volumes/ml_in_action/cv_clf/files/intel_image_clf/valid_imgs_main.delta`

# COMMAND ----------


