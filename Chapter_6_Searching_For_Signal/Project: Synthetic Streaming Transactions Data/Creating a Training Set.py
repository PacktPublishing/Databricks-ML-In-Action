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
      lookup_key=["CustomerID","eventTimestamp"],
      feature_names=["transactionCount", "isTimeout"]
    ),
    FeatureLookup(
      table_name="product_3hour_max_price_ft",
      lookup_key=['Product','TransactionHour']
    ),
    FeatureFunction(
      udf_name="product_difference_ratio_on_demand_feature",
      input_bindings={"max_price":"n1", "transaction_amount":"n2"},
      output_name="max_difference_ratio"
    ),
]

# COMMAND ----------

# DBTITLE 1,Create the training set
training_set = fe.create_training_set(
    df=spark.table("hive_metastore.lakehouse_prod_in_action.synthetic_transactions"),
    feature_lookups=training_feature_lookups,
    label=label_name,
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)
