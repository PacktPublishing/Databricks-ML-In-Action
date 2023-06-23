# Databricks notebook source
# MAGIC %md
# MAGIC https://www.kaggle.com/competitions/store-sales-time-series-forecasting

# COMMAND ----------

# MAGIC %run ./../setup

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup
 
fs = FeatureStoreClient()
model_name = "favorita_sales_forecasting"
primary_keys=["r"],

# COMMAND ----------

# MAGIC %sql
# MAGIC USE lakehouse_in_action;
# MAGIC SHOW TABLES LIKE 'favorita*'

# COMMAND ----------

df = sql("SELECT * FROM favorita_stores s LEFT JOIN favorita_holiday_events h ON s.city == h.locale_name")

display(df)

# COMMAND ----------

df = sql("""
SELECT t.*, o.dcoilwtico as oil_ten_day_lag, s.city, s.state, s.`cluster` as store_cluster, s.type as store_type 
FROM favorita_train_set t 
LEFT JOIN favorita_stores s ON t.store_nbr == s.store_nbr
RIGHT JOIN favorita_oil o ON date(t.`date`) == (date(o.`date`)+10)
""")

autoMLdf = df.where(df.date >= '2016-01-01').drop(df.id)
autoMLdf.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable("favorita_autoML")

# COMMAND ----------

display(autoMLdf)
#comparing to be sure my lag worked as expected

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT date(`date`), date(`date`)+10 as lag, dcoilwtico FROM favorita_oil WHERE `date` >= '2017-01-01'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM favorita_test_set

# COMMAND ----------



# COMMAND ----------

df = sql(
    """
SELECT
  `date`,
  store_nbr,
  sum(sales) as aggSales
FROM
  favorita_train_set
GROUP BY
  `date`,
  store_nbr
ORDER BY
  `date`
  """)
df.createOrReplaceTempView("aggView")

aggAutoML = sql(
    """SELECT t.*, o.dcoilwtico as oil_ten_day_lag, s.city, 
        s.state,s.`cluster` as store_cluster,s.type as store_type FROM aggView t 
LEFT JOIN favorita_stores s ON t.store_nbr == s.store_nbr
RIGHT JOIN favorita_oil o ON date(t.`date`) == (date(o.`date`)+10)
"""
)

aggAutoML = aggAutoML.where(aggAutoML.date >= "2016-01-01")
print(aggAutoML.columns)
aggAutoML.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable(
    "favorita_autoML_agg"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   `date`,
# MAGIC   store_nbr,
# MAGIC   sum(sales) as aggSales
# MAGIC FROM
# MAGIC   favorita_train_set
# MAGIC GROUP BY
# MAGIC   `date`,
# MAGIC   store_nbr
# MAGIC ORDER BY
# MAGIC   `date`

# COMMAND ----------



# COMMAND ----------

favorita_transactions = sql("SELECT * FROM favorita_transactions")

# Creates a time-series feature table for the transaction data using the store_nbr as a primary key and the date as the timestamp key. 
fs.create_table(
    f"{database_name}.fs_favorita_transactions",
    primary_keys=["store_nbr"],
    timestamp_keys=["date"],
    df=favorita_transactions,
    description="The number of transaction per day per store",
)

# COMMAND ----------

favorita_holiday_events = sql("SELECT * FROM favorita_holiday_events")

# Creates a time-series feature table for the transaction data using the date and locale_name as the primary keys. 
fs.create_table(
    f"{database_name}.fs_favorita_holiday_events",
    primary_keys=["date","type","locale_name","description"],
    df=favorita_holiday_events,
    description="The hoildays",
)

# COMMAND ----------

favorita_stores = sql("SELECT * FROM favorita_stores")

fs.create_table(
    f"{database_name}.fs_favorita_stores",
    primary_keys=["store_nbr"],
    df=favorita_stores,
    description="Store data",
)

# COMMAND ----------

favorita_oil = sql("SELECT * FROM favorita_oil")

fs.create_table(
    f"{database_name}.fs_favorita_oil",
    primary_keys=["date"],
    df=favorita_oil,
    description="proxy for economy",
)

# COMMAND ----------

favorita_train_set = sql("SELECT * FROM favorita_train_set")

fs.create_table(
    f"{database_name}.fs_favorita_train_set",
    primary_keys=["store_nbr","family"],
    timestamp_keys=["date"],
    df=favorita_train_set,
    description="training data",
)

# COMMAND ----------

favorita_train_set = sql("SELECT year(date), count(*) FROM favorita_train_set GROUP BY year(date)")
display(favorita_train_set)
# fs.create_table(
#     f"{database_name}.fs_favorita_train_set",
#     primary_keys=["store_nbr","family"],
#     timestamp_keys=["date"],
#     df=favorita_train_set,
#     description="training data",
# )

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

df = ps.read_csv(r'/folder-of-csv-files/', sep=',', decimal='.')
df.columns = ['a','b','c']
df['a2'] = df.a * df.a
df.to_table("pyspark_pandas_as_table")

# COMMAND ----------

import pandas as pd

df = pd.read_csv(r'/folder/single.csv', sep=',', decimal='.')
df.columns = ['a', 'b', 'c']
df['a2'] = df.a * df.a
spark.createDataFrame(df).write.saveAsTable("pandas_as_table")

# COMMAND ----------




# COMMAND ----------

import pyspark.pandas as ps

df = ps.read_csv(r'/folder-of-csv-files/', sep=',', decimal='.')
df.columns = ['a','b','c']
df['a2'] = df.a * df.a
df.to_table("pyspark_pandas_as_table")
