# Databricks notebook source
# MAGIC %md
# MAGIC ## Favorita Forecasting
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Import & Initialize the Feature Store
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup
 
fs = FeatureStoreClient()
model_name = "favorita_sales_forecasting"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reviewing the data

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES LIKE 'favorita*'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   favorita_holiday_events

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM favorita_stores

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

# DBTITLE 1,Connecting local and regional holidays to the respective stores
df = sql(
"""
    SELECT
      h.`date`,
      h.`type` as holiday_type,
      h.locale,
      s.*,
      s.`type` as store_type
    FROM
      favorita_stores s
      INNER JOIN favorita_holiday_events h ON (
        s.city == h.locale_name
        OR s.state == h.locale_name
      )
"""
)

display(df)
df = df.drop("type")
fs.create_table(
    f"{database_name}.fs_stores_local_regional_holidays",
    primary_keys=["date", "store_nbr","locale","holiday_type"],
    df=df,
    description="Joined stores with Holidays by locale_name = city or locale_name = state. National holidays are not included.",
)

# COMMAND ----------

# DBTITLE 1,Some dates have more than one national holiday
# MAGIC %sql
# MAGIC SELECT * FROM favorita_holiday_events WHERE `date`==date('2016-05-07T00:00:00.000+0000')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE favorita_national_holidays AS (
# MAGIC   SELECT
# MAGIC     `date`,
# MAGIC     MIN(`type`) as holiday_type
# MAGIC   FROM
# MAGIC     favorita_holiday_events
# MAGIC   WHERE
# MAGIC     locale == "National"
# MAGIC   GROUP BY
# MAGIC     ALL
# MAGIC   ORDER BY
# MAGIC     `date`
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   t.`date`,
# MAGIC   t.store_nbr,
# MAGIC   t.family,
# MAGIC   t.onpromotion,
# MAGIC   count(1) as cunt
# MAGIC FROM
# MAGIC   favorita_train_set t
# MAGIC   LEFT JOIN favorita_national_holidays h ON (date(t.`date`) == date(h.`date`))
# MAGIC   GROUP BY ALL
# MAGIC ORDER BY
# MAGIC   cunt DESC

# COMMAND ----------

# DBTITLE 1,National holidays joined to the training data
df = sql("""
      SELECT
        t.`date`,
        t.store_nbr,
        t.family,
        t.onpromotion,
        IFNULL(h.holiday_type,"None") as national_holiday_type,
        t.sales
      FROM
        favorita_train_set t
        LEFT JOIN favorita_national_holidays h ON (date(h.`date`) == date(t.`date`))
      ORDER BY
        t.`date`
      """)

fs.create_table(
    f"{database_name}.fs_train_national_holidays",
    primary_keys=["date", "store_nbr","family","onpromotion"],
    df=df,
    description="Joined the training dataset with National holidays only.",
)

# COMMAND ----------

# DBTITLE 1,Lagging the oil price for a proxy of the economy
df = sql(
    """
    SELECT
      date(`date`),
      date(`date`) -10 as join_on_date,
      dcoilwtico as lag10_oil_price
    FROM
      favorita_oil
  """
)

fs.create_table(
    f"{database_name}.fs_oil_lag",
    primary_keys=["join_on_date"],
    df=df,
    description="The lag10_oil_price is the price of oil 10 days after the join_on_date.",
)
