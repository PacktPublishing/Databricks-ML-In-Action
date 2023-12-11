# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 4: Exploring and Cleaning Toward a Silver Layer
# MAGIC
# MAGIC ## Favorita Forecasting - Oil Data Imputing
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# DBTITLE 1,View oil prices data
# MAGIC %sql
# MAGIC SELECT * FROM oil_prices

# COMMAND ----------

# MAGIC %md Each day has multiple prices, so we will need to index the data.

# COMMAND ----------

# DBTITLE 1,Read oil price data into a spark dataframe
import pyspark.pandas as ps

df = ps.read_table("oil_prices", index_col="date")

# COMMAND ----------

# MAGIC %md We are going to use the reindex() command to create an updated index based on the minimum and maximum of the oil price data, and changes null values to NaN. 

# COMMAND ----------

df.reindex(ps.date_range(df.index.min(), 
                         df.index.max())
          ).reset_index().rename(columns={'index':'date'}).head(20)


# COMMAND ----------

# MAGIC %md
# MAGIC We are going to use reindex functions and it's filling functions to create a dataframe that contains each day ones with a Nan as the oil index value when the data wasn't in the original oil data set.

# COMMAND ----------

# DBTITLE 1,Write dataframe with imputed data to silver table
df = df.reindex(ps.date_range(df.index.min(), df.index.max())).reset_index().rename(columns={'index':'date'}).ffill()
df.to_table('oil_prices_silver')
