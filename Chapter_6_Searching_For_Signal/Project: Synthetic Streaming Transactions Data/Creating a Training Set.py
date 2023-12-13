# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ##Synthetic data - Creating a Training Set

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM transaction_count_history LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM raw_transactions LIMIT 10

# COMMAND ----------

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
      table_name="product_3hour_max_price_ft",
      rename_outputs={
          "LookupTimestamp": "TransactionTimestamp"
        },
      lookup_key=['Product'],
      timestamp_lookup_key='TransactionTimestamp'
    )
]

inference_feature_lookups = [
    FeatureLookup(
      table_name="transaction_count_ft",
      rename_outputs={
          "eventTimestamp": "TransactionTimestamp"
        },
      lookup_key=["CustomerID","TransactionTimestamp"],
      feature_names=["transactionCount", "isTimeout"],
      timestamp_lookup_key = "TransactionTimestamp"
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"MaxProductAmount", "transaction_amount":"Amount"},
      output_name="MaxDifferenceRatio"
    )
]

# COMMAND ----------

# DBTITLE 1,Create the training set
raw_transactions_df = spark.table("raw_transactions")

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
)
training_df = training_set.load_df()

# COMMAND ----------

# DBTITLE 1,Create the training set
raw_transactions_df = spark.table("raw_transactions")

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------

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
      table_name="product_3hour_max_price_ft",
      rename_outputs={
          "LookupTimestamp": "TransactionTimestamp"
        },
      lookup_key=['Product'],
      timestamp_lookup_key='TransactionTimestamp'
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"MaxProductAmount", "transaction_amount":"Amount"},
      output_name="max_difference_ratio"
    )
]

inference_feature_lookups = [
    FeatureLookup(
      table_name="transaction_count_ft",
      rename_outputs={
          "eventTimestamp": "TransactionTimestamp"
        },
      lookup_key=["CustomerID","TransactionTimestamp"],
      feature_names=["transactionCount", "isTimeout"],
      timestamp_lookup_key = "TransactionTimestamp"
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"MaxProductAmount", "transaction_amount":"Amount"},
      output_name="MaxDifferenceRatio"
    )
]

# COMMAND ----------

# DBTITLE 1,Create the training set
raw_transactions_df = spark.table("raw_transactions")

training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------


