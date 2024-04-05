# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ##Streaming data - Creating a Maximum Price Feature Table

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Step 1: Calculate maximum price per product and save as a feature table

# COMMAND ----------

sql("DROP TABLE IF EXISTS product_3minute_max_price_ft")
raw_transactions_df = spark.table("raw_transactions")

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

# Create feature table with Product as the primary key
# We're using the convention of appending feature table names with "_ft"
fe.create_table(
  df=max_price_df,
  name='product_3minute_max_price_ft',
  primary_keys=['Product','LookupTimestamp'],
  timeseries_columns='LookupTimestamp',
  schema=max_price_df.schema,
  description="Maximum price per product over the last 3 minutes for Synthetic Transactions. Join on TransactionTimestamp to get the max product price from last minute's 3 minute rolling max"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create a Python UDF to calculate the difference ratio of each transaction

# COMMAND ----------

# DBTITLE 1,Creating a Python UDF and saving to Unity Catalog.
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION product_difference_ratio_on_demand_feature(max_price FLOAT, transaction_amount FLOAT)
# MAGIC RETURNS float
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Calculate the difference ratio for a product at time of transaction (maximum price - transaction amount)/maximum price.'
# MAGIC AS $$
# MAGIC def calc_ratio_difference(n1: float, n2: float) -> float:
# MAGIC   return round(((n1 - n2)/n1),2)
# MAGIC
# MAGIC return calc_ratio_difference(max_price, transaction_amount)
# MAGIC $$

# COMMAND ----------

# DBTITLE 1,Testing out the function.
# MAGIC %sql
# MAGIC select product_difference_ratio_on_demand_feature(15.01, 100.67) as difference_ratio
