# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Searching for Signal
# MAGIC
# MAGIC ## Synthetic data - Creating a training set

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating the training set
# MAGIC
# MAGIC With the timeserires features, the first values of the features will be later than the initial raw transactions. We start with determining the earliest raw transaction we want to include in the training set. For the on-demand UDF, nulls will throw an ugly error. For the transaction count, we simply won't have the feature value. Therefore we must have a value for the max product, but not necessarily for the transaction count. We will make sure both have values.

# COMMAND ----------

display(sql("SELECT MIN(lookupTimestamp) as ts FROM product_3minute_max_price_ft"))

# COMMAND ----------

display(sql("SELECT MIN(eventTimestamp) as ts FROM transaction_count_history"))

# COMMAND ----------

# DBTITLE 1,Define the feature lookups
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
fe = FeatureEngineeringClient()

training_feature_lookups = [
    FeatureLookup(
      table_name="transaction_count_history",
      rename_outputs={
          "eventTimestamp": "TransactionTimestamp"
        },
      lookup_key=["CustomerID"],
      feature_names=["transactionCount", "isTimeout"],
      timestamp_lookup_key = "TransactionTimestamp"
    ),
    FeatureLookup(
      table_name="product_3minute_max_price_ft",
      rename_outputs={
          "LookupTimestamp": "TransactionTimestamp"
        },
      lookup_key=['Product'],
      
      timestamp_lookup_key='TransactionTimestamp'
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"MaxProductAmount", "transaction_amount":"Amount"},
      output_name="MaxDifferenceRatio"
    )
]

# COMMAND ----------

# DBTITLE 1,Create the training set
raw_transactions_df = sql("SELECT * FROM raw_transactions WHERE timestamp(TransactionTimestamp) > timestamp('2024-02-16T23:18:01.080+00:00')")

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
    exclude_columns="_rescued_data"
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)
