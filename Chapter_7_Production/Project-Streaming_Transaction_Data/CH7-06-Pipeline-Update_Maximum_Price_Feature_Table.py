# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ##Synthetic Transactions data - Creating a Maximum Price Feature Table

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Step 1: Calculate maximum price per product and save as a feature table

# COMMAND ----------

raw_transactions_df = sql("SELECT Amount,CustomerID,Label,Product,TransactionTimestamp FROM prod_raw_transactions rt INNER JOIN (SELECT MAX(LookupTimestamp) as max_timestamp FROM product_3minute_max_price_ft) ts ON rt.TransactionTimestamp > (ts.max_timestamp - INTERVAL 1 MINUTE)")

# COMMAND ----------

# DBTITLE 1,Creating a feature table of product maximum prices.
from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.sql.functions as F 

fe = FeatureEngineeringClient()

time_window = F.window(
    F.col("TransactionTimestamp"),
    windowDuration="3 minutes",
    slideDuration="1 minute",
).alias("time_window")
              
max_price_df = (
  raw_transactions_df
    .groupBy(F.col("Product"),time_window)
    .agg(F.max(F.col("Amount")).cast("float").alias("MaxProductAmount"))
    .withColumn("LookupTimestamp", 
                F.date_trunc('minute',
                          F.col("time_window.end") + F.expr('INTERVAL 1 MINUTE')))
    .drop("time_window")
)

# Update feature table with Product as the primary key
# We're using the convention of appending feature table names with "_ft"
try:
  fe.create_table(
    df=max_price_df,
    name='product_3minute_max_price_ft',
    primary_keys=['Product','LookupTimestamp'],
    timeseries_columns='LookupTimestamp',
    schema=max_price_df.schema,
    description="Maximum price per product over the last 3 minutes for Synthetic Transactions. Join on TransactionTimestamp to get the max product price from last minute's 3 minute rolling max"
    )
except:
  fe.write_table(
    df=max_price_df,
    name='product_3minute_max_price_ft'
  )

# COMMAND ----------

# MAGIC %md
# MAGIC We want to sync this table to the Databricks Online Store. Triggered (or Continuous) sync mode, the source table must have Change data feed enabled.

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `ml_in_action`.`synthetic_transactions`.`product_3minute_max_price_ft` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

dbutils.fs.rm('/Volumes/ml_in_action/synthetic_transactions/files/{table_name}',True)
