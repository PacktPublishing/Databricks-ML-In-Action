# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 4: Getting to Know Your Data
# MAGIC
# MAGIC ## Favorita Forecasting - Oil Data Imputing
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# DBTITLE 1,Read oil price data into a spark dataframe
import pyspark.pandas as ps

df = ps.read_table("oil_prices", index_col="date")
df.head(10)

# COMMAND ----------

# DBTITLE 1,Write dataframe with imputed data to silver table
df = (
    df.reindex(ps.date_range(df.index.min(), df.index.max()))
    .reset_index()
    .rename(columns={"index": "date"})
    .ffill()
)
df.to_table("oil_prices_silver")
df.head(10)
