# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 3: Building out our Bronze Layer
# MAGIC
# MAGIC ## Favorita Sales - Loading Sales CSV data as Delta
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md ##Run setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Set the data path and displaying the files
raw_data_path = volume_data_path + '/raw_data/store-sales-time-series-forecasting/'
display(dbutils.fs.ls(raw_data_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transforming CSV to Delta

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# DBTITLE 1,Holiday Events
df = pd.read_csv(f'{raw_data_path}/holidays_events.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("holiday_events")
display(df)

# COMMAND ----------

# DBTITLE 1,Oil Prices
df = pd.read_csv(f'{raw_data_path}/oil.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("oil_prices")
display(df)

# COMMAND ----------

# DBTITLE 1,Stores
df = pd.read_csv(f'{raw_data_path}/stores.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_stores")
display(df)

# COMMAND ----------

# DBTITLE 1,Test Set
df = pd.read_csv(f'{raw_data_path}/test.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("test_set")
display(df)

# COMMAND ----------

# DBTITLE 1,Train Set
df = pd.read_csv(f'{raw_data_path}/train.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("train_set")
display(df)

# COMMAND ----------

# DBTITLE 1,Transactions
df = pd.read_csv(f'{raw_data_path}/transactions.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_transactions")
display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------


