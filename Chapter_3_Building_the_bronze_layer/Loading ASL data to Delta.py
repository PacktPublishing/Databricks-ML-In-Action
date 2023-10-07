# Databricks notebook source
# MAGIC %md
# MAGIC # ASL Fingerspelling Data to Delta
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
# MAGIC
# MAGIC ##Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

display(dbutils.fs.ls(cloud_storage_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Transforming CSV to Delta

# COMMAND ----------

# MAGIC %md We are loading the CSV data into Delta using the Pandas library because the dataset is small and can be handled easily. 

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.read_csv(f'{cloud_storage_path}/supplemental_metadata.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("supplemental_metadata")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_storage_path}/train.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("training_metadata")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utilizing Volumes
# MAGIC
# MAGIC Let's grant access to Hayley so she can work on the data we are putting in our volume. 
# MAGIC We copy the json file to our Volume so we can access it, as JSON, easily during training and inference. Additionally, we read it in as a dataframe for inspection.

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT WRITE VOLUME
# MAGIC ON VOLUME lakehouse_in_action.asl_fingerspelling.asl_volume
# MAGIC TO `hayley.horn@databricks.com`;

# COMMAND ----------

dbutils.fs.cp(f"{cloud_storage_path}/character_to_prediction_index.json", volume_data_path)

# COMMAND ----------

import pyspark.pandas as psp
data = psp.read_json(f'{cloud_storage_path}/character_to_prediction_index.json')
dic = data.to_dict()
char_2_pred_index = pd.DataFrame([(key,value[0]) for key, value in dic.items()], columns=["char","pred_index"])

# COMMAND ----------

display(char_2_pred_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Transforming Parquet to Delta
# MAGIC
# MAGIC We use regular spark for reading in the parquet files

# COMMAND ----------

display(dbutils.fs.ls(cloud_storage_path+"/supplemental_landmarks/"))

# COMMAND ----------

df = spark.read.parquet(f'{cloud_storage_path}/supplemental_landmarks/')
df.write.mode("overwrite").saveAsTable("supplemental_landmarks")
display(df)

# COMMAND ----------

df = spark.read.parquet(f'{cloud_storage_path}/train_landmarks/')
df.write.mode("overwrite").saveAsTable("train_landmarks")
display(df)

# COMMAND ----------


