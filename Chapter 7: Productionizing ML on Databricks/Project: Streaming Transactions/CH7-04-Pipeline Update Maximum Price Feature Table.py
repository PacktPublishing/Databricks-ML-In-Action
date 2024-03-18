# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Productionizing ML on Databricks
# MAGIC
# MAGIC ##Transaction data - Creating a Maximum Price Feature Table

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $env=prod

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Calculate maximum price per product and save as a feature table

# COMMAND ----------

table_name = "raw_transactions"
ft_name = "product_3minute_max_price_ft"

if not spark.catalog.tableExists(ft_name) or spark.table(tableName=ft_name).isEmpty():
  raw_transactions_df = sql(f"SELECT Amount,CustomerID,Label,Product,TransactionTimestamp FROM {table_name}")
else:  
  raw_transactions_df = sql(f"SELECT Amount,CustomerID,Label,Product,TransactionTimestamp FROM {table_name} rt INNER JOIN (SELECT MAX(LookupTimestamp) as max_timestamp FROM {ft_name}) ts ON rt.TransactionTimestamp > (ts.max_timestamp - INTERVAL 1 MINUTE)")

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
if not spark.catalog.tableExists(f'{ft_name}'):
  fe.create_table(
    df=max_price_df,
    name=f'{ft_name}',
    primary_keys=['Product','LookupTimestamp'],
    timeseries_columns='LookupTimestamp',
    schema=max_price_df.schema,
    description="Maximum price per product over the last 3 minutes for Synthetic Transactions. Join on TransactionTimestamp to get the max product price from last minute's 3 minute rolling max"
    )
else:
  fe.write_table(
    df=max_price_df,
    mode='merge',
    name=f'{ft_name}'
  )

# COMMAND ----------

display(max_price_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Python UDF to calculate the difference ratio of each transaction

# COMMAND ----------

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

# MAGIC %sql
# MAGIC select product_difference_ratio_on_demand_feature(15.01, 100.67) as difference_ratio

# COMMAND ----------


