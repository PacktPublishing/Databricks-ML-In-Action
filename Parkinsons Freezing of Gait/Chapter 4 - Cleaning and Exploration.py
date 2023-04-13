# Databricks notebook source
# MAGIC %run ./setup

# COMMAND ----------

# MAGIC %md Fields on parkinsons_defog_train table <br> 
# MAGIC Time
# MAGIC AccV
# MAGIC AccML
# MAGIC AccAP
# MAGIC StartHesitation
# MAGIC Turn
# MAGIC Walking
# MAGIC Valid
# MAGIC Task

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 

df = sql('select * from lakehouse_in_action.parkinsons_train_defog where Valid == TRUE AND Task == TRUE AND (StartHesitation + Turn + Walking) > 0')

df_index = df.select("*").withColumn("id", monotonically_increasing_id())
display(df_index)
df_index.write.mode("overwrite").saveAsTable("unambiguous_indexed")

# COMMAND ----------

# MAGIC %md We could use bamboolib here, or we could use the pandas profiler

# COMMAND ----------


