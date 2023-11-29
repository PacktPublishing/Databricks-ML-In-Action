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
from pyspark.sql.functions import max
from pyspark.sql.functions import to_date

fe = FeatureEngineeringClient()

max_price_df = (
    raw_transactions_df.withColumn("TransactionDate", to_date("TransactionTimestamp"))
    .groupBy("Product", "TransactionDate")
    .agg(max("Amount").alias("MaxProductAmount"))
)

# Create feature table with Product as the primary key
# We're using the convention of appending feature table names with "_ft"
customer_feature_table = fe.create_table(
  df=max_price_df,
  name='lakehouse_in_action.synthetic_transactions.product_max_price_ft',
  primary_keys=['Product','TransactionDate'],
  schema=max_price_df.schema,
  description='Maximum price per day for Synthetic Transactions products.'
)

# COMMAND ----------

display(max_price_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create a Python UDF to calculate the discount of each transaction

# COMMAND ----------

# DBTITLE 1,Creating a Python UDF and saving to Unity Catalog.
# MAGIC %sql
# MAGIC CREATE FUNCTION IF NOT EXISTS product_discount_on_demand_feature(max_price FLOAT, transaction_amount FLOAT)
# MAGIC RETURNS float
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Calculate the percent discount for a product at time of transaction (maximum price - transaction amount)/maximum price.'
# MAGIC AS $$
# MAGIC def calc_discount(n1: float, n2: float) -> float:
# MAGIC   return round(((n1 - n2)/n1),2)
# MAGIC
# MAGIC return calc_discount(max_price, transaction_amount)
# MAGIC $$

# COMMAND ----------

# DBTITLE 1,Testing out the function.
# MAGIC %sql
# MAGIC select product_discount_on_demand_feature(15, 100) as discount

# COMMAND ----------


