# Databricks notebook source
# MAGIC %md
# MAGIC # Favorita Sales
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
# MAGIC
# MAGIC ##Run setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

display(dbutils.fs.ls(spark_storage_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transforming CSV to Delta

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# DBTITLE 1,Holiday Events
df = pd.read_csv(f'{cloud_storage_path}/holidays_events.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("holiday_events")
display(df)

# COMMAND ----------

# DBTITLE 1,Oil Prices
df = pd.read_csv(f'{cloud_storage_path}/oil.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("oil_prices")
display(df)

# COMMAND ----------

# DBTITLE 1,Stores
df = pd.read_csv(f'{cloud_storage_path}/stores.csv', sep=',', decimal='.')
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_stores")
display(df)

# COMMAND ----------

# DBTITLE 1,Test Set
df = pd.read_csv(f'{cloud_storage_path}/test.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("test_set")
display(df)

# COMMAND ----------

# DBTITLE 1,Train Set
df = pd.read_csv(f'{cloud_storage_path}/train.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("train_set")
display(df)

# COMMAND ----------

# DBTITLE 1,Transactions
df = pd.read_csv(f'{cloud_storage_path}/transactions.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("favorita_transactions")
display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC USE `lakehouse_in_action`.`favorita_forecasting`

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------


