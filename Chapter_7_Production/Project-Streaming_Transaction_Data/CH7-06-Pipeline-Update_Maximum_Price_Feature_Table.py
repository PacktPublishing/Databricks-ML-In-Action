# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ##Synthetic Transactions data - Creating a Maximum Price Feature Table

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

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
fe.write_table(
  df=max_price_df,
  name='product_3minute_max_price_ft'
)
