# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 4: Exploring and cleaning toward the silver layer
# MAGIC
# MAGIC ## Synthetic data - Exploration
# MAGIC

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize the streaming table

# COMMAND ----------

# DBTITLE 1,Simple select allows many visualizations
# MAGIC %sql
# MAGIC select * from synthetic_transactions_dlt

# COMMAND ----------


