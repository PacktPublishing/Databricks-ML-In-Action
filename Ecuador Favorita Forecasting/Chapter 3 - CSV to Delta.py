# Databricks notebook source
# MAGIC %md
# MAGIC https://www.kaggle.com/competitions/store-sales-time-series-forecasting

# COMMAND ----------

# MAGIC %run ./setup

# COMMAND ----------

import pandas as pd
import pyspark.pandas as ps

spark_data_path = 'dbfs:/FileStore/LakehouseInAction/store-sales-time-series-forecasting/'
cloud_data_path = '/dbfs/FileStore/LakehouseInAction/store-sales-time-series-forecasting/'

# COMMAND ----------

dbutils.fs.ls(spark_data_path)

# COMMAND ----------

# DBTITLE 1,Transforming CSV to Delta
df = pd.read_csv(f'{cloud_data_path}holidays_events.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
spark.createDataFrame(df).write.saveAsTable("favorita_holiday_events")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_data_path}oil.csv', sep=',', decimal='.')
df['date'] = pd.to_datetime(df['date'])
#spark.createDataFrame(df).write.saveAsTable("favorita_oil")
display(df)

# COMMAND ----------

df = pd.read_csv(f'{cloud_data_path}stores.csv', sep=',', decimal='.')
#spark.createDataFrame(df).write.saveAsTable("favorita_stores)
display(df)

# COMMAND ----------

df = ps.read_csv(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/defog/', sep=',', decimal='.')
display(df)
df.to_table("parkinsons_train_defog")

# COMMAND ----------

df = ps.read_csv(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/notype/', sep=',', decimal='.')
display(df)
df.to_table("parkinsons_train_notype")

# COMMAND ----------

df = ps.read_csv(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/', sep=',', decimal='.')
display(df)
df.to_table("parkinsons_train_tdcsfog")

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/')

# COMMAND ----------

df = ps.read_csv(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/', sep=',', decimal='.')
display(df)
df.to_table("parkinsons_test_tdcsfog")

# COMMAND ----------

df = ps.read_csv(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/test/defog/', sep=',', decimal='.')
display(df)
df.to_table("parkinsons_test_defog")

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/unlabeled/')

# COMMAND ----------

df = ps.read_parquet(r'/FileStore/LakehouseInAction/tlvmc-parkinsons-freezing-gait-prediction/unlabeled/')
display(df)
df.to_table("parkinsons_unlabeled")

# COMMAND ----------


