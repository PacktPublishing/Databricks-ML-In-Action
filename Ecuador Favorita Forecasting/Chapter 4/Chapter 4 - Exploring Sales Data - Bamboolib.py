# Databricks notebook source
# MAGIC %md
# MAGIC https://www.kaggle.com/competitions/store-sales-time-series-forecasting

# COMMAND ----------

# MAGIC %run ./../setup

# COMMAND ----------

# MAGIC %sql
# MAGIC USE lakehouse_in_action;
# MAGIC SHOW TABLES LIKE 'favorita*'

# COMMAND ----------

import bamboolib as bam
import pandas as pd

df = spark.table("hive_metastore.lakehouse_in_action.favorita_transactions").sample(fraction=.2).toPandas()
df

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Does every store have the same number of data points?
import pandas as pd; import numpy as np
# Remember to load the non-limited dataframe
df = spark.table("hive_metastore.lakehouse_in_action.favorita_transactions").toPandas()

# Step: Group by store_nbr and calculate new column(s)
store_row_count = df.groupby(['store_nbr']).agg(row_count=('date', 'size')).reset_index()

# Visualize
import plotly.express as px
fig = px.bar(store_row_count, x='store_nbr', y='row_count', color='store_nbr')
fig

# COMMAND ----------

# DBTITLE 1,Filter shows 19% of the data is missing dates
num_stores = store_row_count.size
# Step: Keep rows where row_count < 1670
store_low_row_count = store_row_count.loc[store_row_count['row_count'] < 1600]
# Filter: removed 44 rows (81%)
num_low_stores =store_low_row_count.size
print(f'{num_low_stores} out of {num_stores} were filtered, thats {round(num_low_stores*1.0/num_stores*100)}% of stores')

# COMMAND ----------

# Step: Keep rows where store_nbr == 53
low_row_count_transactions = df.loc[df['store_nbr'].isin(store_low_row_count['store_nbr'])]
lrct = low_row_count_transactions.pivot(index=['date'], columns='store_nbr', values='transactions')
lrct.reset_index(inplace=True)
lrct.fillna(0, inplace=True)

# COMMAND ----------

display(lrct)

# COMMAND ----------

from pyspark.sql.functions import col 
favorita_holiday_events = sql("SELECT * FROM lakehouse_in_action.favorita_holiday_events")
display(favorita_holiday_events.groupBy(["date","locale_name"]).count())
display(favorita_holiday_events.select("*").where(col("date")=="2016-05-08"))

# COMMAND ----------

ddf = sql("SELECT * FROM lakehouse_in_action.favorita_train_set")
display(ddf)

# COMMAND ----------

ddf = sql("SELECT * FROM lakehouse_in_action.favorita_transactions")
display(ddf)

# COMMAND ----------


