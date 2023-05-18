# Databricks notebook source
# MAGIC %md
# MAGIC https://www.kaggle.com/competitions/store-sales-time-series-forecasting

# COMMAND ----------

# MAGIC %run ./../setup

# COMMAND ----------

import pandas as pd

spark_data_path = 'dbfs:/FileStore/LakehouseInAction/store-sales-time-series-forecasting/'
cloud_data_path = '/dbfs/FileStore/LakehouseInAction/store-sales-time-series-forecasting/'

# COMMAND ----------

dbutils.fs.ls(spark_data_path)

# COMMAND ----------

# DBTITLE 1,Transforming CSV to Delta
df = pd.read_csv(f'{cloud_data_path}holidays_events.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_holiday_events")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_data_path}oil.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_oil")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_data_path}stores.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_stores")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_data_path}test.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_test_set")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_data_path}train.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_train_set")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_data_path}transactions.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_transactions")
display(df)

# COMMAND ----------


