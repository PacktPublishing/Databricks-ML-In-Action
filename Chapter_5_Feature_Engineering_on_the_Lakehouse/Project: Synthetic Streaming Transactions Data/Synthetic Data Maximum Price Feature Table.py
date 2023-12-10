# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ##Synthetic Transactions data - Creating a Maximum Price Feature Table

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Step 1: Calculate maximum price per product and save as a feature table

# COMMAND ----------

# parametrize
raw_transactions_df = spark.sql("select * from hive_metastore.lakehouse_prod_in_action.synthetic_transactions")

# COMMAND ----------

# DBTITLE 1,Creating a feature table of product maximum prices.
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import max, to_date, col, window, date_trunc, to_timestamp, expr

fe = FeatureEngineeringClient()

time_window = window(
    col("TransactionTimestamp"),
    windowDuration="3 hours",
    slideDuration="1 hour",
).alias("time_window")
              
max_price_df = (
    raw_transactions_df
      .withColumn("TransactionHour", 
                   date_trunc('hour',to_timestamp("TransactionTimestamp")))
      .groupBy(col("Product"),time_window)
      .agg(max(col("Amount")).alias("MaxProductAmount"))
      .withColumn("TransactionHour", 
                  date_trunc('hour',
                            col("time_window.end") + expr('INTERVAL 1 HOUR')))
      .drop("time_window")
)

# Create feature table with Product as the primary key
# We're using the convention of appending feature table names with "_ft"
fe.create_table(
  df=max_price_df,
  name='product_3hour_max_price_ft',
  primary_keys=['Product','TransactionHour'],
  schema=max_price_df.schema,
  description="Maximum price per product over the last 3 hours for Synthetic Transactions. Join on TransactionHour to get the max product price from last hour's 3 hour rolling max"
)

# COMMAND ----------

display(max_price_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create a Python UDF to calculate the difference ratio of each transaction

# COMMAND ----------

# DBTITLE 1,Creating a Python UDF and saving to Unity Catalog.
# MAGIC %sql
# MAGIC CREATE FUNCTION IF NOT EXISTS product_difference_ratio_on_demand_feature(max_price FLOAT, transaction_amount FLOAT)
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
# MAGIC select product_difference_ratio_on_demand_feature(15, 100) as difference_ratio

# COMMAND ----------


