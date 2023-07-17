# Databricks notebook source
# MAGIC %md
# MAGIC ## Favorita Sales
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting

# COMMAND ----------

import pandas as pd

spark_data_path = 'dbfs:/FileStore/LakehouseInAction/store-sales-time-series-forecasting/'
cloud_data_path = '/dbfs/FileStore/LakehouseInAction/store-sales-time-series-forecasting/'

# COMMAND ----------

dbutils.fs.ls(spark_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transforming CSV to Delta

# COMMAND ----------

# DBTITLE 1,Holiday Events
df = pd.read_csv(f'{cloud_data_path}holidays_events.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_holiday_events")
display(df)

# COMMAND ----------

# DBTITLE 1,Oil Prices
df = pd.read_csv(f'{cloud_data_path}oil.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_oil")
display(df)

# COMMAND ----------

# DBTITLE 1,Stores
df = pd.read_csv(f'{cloud_data_path}stores.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_stores")
display(df)

# COMMAND ----------

# DBTITLE 1,Test Set
df = pd.read_csv(f'{cloud_data_path}test.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_test_set")
display(df)

# COMMAND ----------

# DBTITLE 1,Train Set
df = pd.read_csv(f'{cloud_data_path}train.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_train_set")
display(df)

# COMMAND ----------

# DBTITLE 1,Transactions
df = pd.read_csv(f'{cloud_data_path}transactions.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_transactions")
display(df)

# COMMAND ----------


