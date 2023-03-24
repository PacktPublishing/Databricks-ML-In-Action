# Databricks notebook source
# MAGIC %sql
# MAGIC USE DATABASE lakehouse_in_action

# COMMAND ----------

import bamboolib as bam
bam

# COMMAND ----------

df = pd.read_csv(r'/dbfs/FileStore/LakehouseInAction/electric-motor-temperature/measures_v2.csv', sep=',', decimal='.')
# Step: Change data type of profile_id to Categorical/Factor
df['profile_id'] = df['profile_id'].astype('category') #this was ignored by spark
spark.createDataFrame(df).write.saveAsTable("lakehouse_in_action.electric_motor_temp_bronze")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM electric_motor_temp_bronze

# COMMAND ----------


