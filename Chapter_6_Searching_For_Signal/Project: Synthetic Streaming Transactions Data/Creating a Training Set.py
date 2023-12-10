# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 5: Feature Engineering
# MAGIC
# MAGIC ##Synthetic data - Creating a Training Set

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $catalog=lakehouse_in_action

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
fe = FeatureEngineeringClient()

fe.set_feature_table_tag(name="transaction_count_ft", key="FE_role", value="online_serving")
fe.set_feature_table_tag(name="transaction_count_history", key="FE_role", value="training_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM transaction_count_history LIMIT 10

# COMMAND ----------

training_feature_lookups = [
    FeatureLookup(
      table_name="transaction_count_history",
      lookup_key=["CustomerID"],
      timestamp_lookup_key=["eventTimestamp"],
      feature_names=["transactionCount", "isTimeout"]
    ),
    FeatureLookup(
      table_name="product_3hour_max_price_ft",
      lookup_key=['Product','TransactionHour']
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"MaxProductAmount", "transaction_amount":"Amount"},
      output_name="max_difference_ratio"
    ),
]

# COMMAND ----------

from pyspark.sql.functions import date_trunc, to_timestamp
raw_transactions_df = spark.table("lakehouse_in_action.synthetic_transactions.labeled_transactions")
raw_transactions_df = raw_transactions_df.withColumn("TransactionHour", date_trunc('hour',to_timestamp("TransactionTimestamp")))

# COMMAND ----------

# DBTITLE 1,Create the training set
training_set = fe.create_training_set(
    df=raw_transactions_df,
    feature_lookups=training_feature_lookups,
    label="Label",
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------


