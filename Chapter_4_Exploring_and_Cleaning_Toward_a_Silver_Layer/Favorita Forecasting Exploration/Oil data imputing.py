# Databricks notebook source
# MAGIC %run ../../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM oil_prices

# COMMAND ----------

import pyspark.pandas as ps

df = ps.read_table("oil_prices", index_col="date")

# COMMAND ----------

df.reindex(ps.date_range(df.index.min(), 
                         df.index.max())
          ).reset_index().rename(columns={'index':'date'}).head(20)


# COMMAND ----------

df = df.reindex(ps.date_range(df.index.min(), df.index.max())).reset_index().rename(columns={'index':'date'}).ffill()
df.to_table('oil_prices_silver')
