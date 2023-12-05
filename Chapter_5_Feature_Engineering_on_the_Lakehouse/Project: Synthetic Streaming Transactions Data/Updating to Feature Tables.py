# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Feature Engineering
# MAGIC
# MAGIC ##Synthetic data - Updating to Feature Tables

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE transaction_count_ft ALTER COLUMN CustomerID SET NOT NULL;
# MAGIC ALTER TABLE transaction_count_ft ADD PRIMARY KEY(CustomerID);
# MAGIC
# MAGIC ALTER TABLE transaction_count_history ALTER COLUMN CustomerID SET NOT NULL;
# MAGIC ALTER TABLE transaction_count_history ALTER COLUMN eventTimestamp SET NOT NULL;
# MAGIC ALTER TABLE transaction_count_history ADD PRIMARY KEY(CustomerID);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating features using historical data

# COMMAND ----------

df = table("hive_metastore.lakehouse_in_action.synthetic_transactions")
df = (
  # Sum total purchased for 7 days
  travel_purchase_df.withColumn("lookedup_price_7d_rolling_sum",
      F.sum("price").over(w.Window.partitionBy("user_id").orderBy(F.col("ts_l")).rangeBetween(start=-(7 * 86400), end=0))
  )
  # counting number of purchases per week
  .withColumn("lookups_7d_rolling_sum", 
      F.count("*").over(w.Window.partitionBy("user_id").orderBy(F.col("ts_l")).rangeBetween(start=-(7 * 86400), end=0))
  )
  # total price 7d / total purchases for 7 d 
  .withColumn("mean_price_7d",  F.col("lookedup_price_7d_rolling_sum") / F.col("lookups_7d_rolling_sum"))
    # converting True / False into 1/0
  .withColumn("tickets_purchased", F.col("purchased").cast('int'))
  # how many purchases for the past 6m
  .withColumn("last_6m_purchases", 
      F.sum("tickets_purchased").over(w.Window.partitionBy("user_id").orderBy(F.col("ts_l")).rangeBetween(start=-(6 * 30 * 86400), end=0))
  )
  .select("user_id", "ts", "mean_price_7d", "last_6m_purchases", "user_longitude", "user_latitude"))


# COMMAND ----------

display(df)

# COMMAND ----------


