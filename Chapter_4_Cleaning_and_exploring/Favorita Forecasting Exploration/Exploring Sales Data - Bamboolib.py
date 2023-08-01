# Databricks notebook source
# MAGIC %md
# MAGIC https://www.kaggle.com/competitions/store-sales-time-series-forecasting

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES LIKE 'favorita*'

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

import bamboolib as bam
import pandas as pd

df = spark.table("favorita_train_set").sample(fraction=.2).toPandas()
df

# COMMAND ----------

import pyspark.pandas as ps
df = spark.table("favorita_train_set")
# Keep rows where store_nbr == 44
df44 = df[df['store_nbr'] == 44]
df44.write.mode("overwrite").saveAsTable("favorita_store_44")

# COMMAND ----------

from pyspark.sql.functions import col 
favorita_holiday_events = sql("SELECT * FROM favorita_holiday_events")
display(favorita_holiday_events.groupBy(["locale_name"]).count())

# COMMAND ----------


