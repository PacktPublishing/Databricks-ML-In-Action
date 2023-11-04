# Databricks notebook source
# MAGIC %md
# MAGIC ## Favorita Forecasting
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Import & Initialize the Feature Store
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reviewing the data

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM holiday_events LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM favorita_stores LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering
# MAGIC Lets work on transforming our data
# MAGIC
# MAGIC #### Stores

# COMMAND ----------

# DBTITLE 1,Store features
df = sql(
"""
    SELECT
      *,
      `type` as store_type
    FROM
      favorita_stores
"""
)

df = df.drop("type")
display(df)

fe.create_table(
    name=f"{database_name}.stores_ft",
    primary_keys=["store_nbr"],
    df=df,
    description="Store features include geography, cluster, and store_type.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Holidays

# COMMAND ----------

# DBTITLE 1,Some dates have more than one national holiday
# MAGIC %sql
# MAGIC SELECT * FROM holiday_events WHERE `date`==date('2016-05-07T00:00:00.000+0000')

# COMMAND ----------

# DBTITLE 1,Local holidays
sql("""
  CREATE OR REPLACE TABLE local_holidays_bronze AS (
    SELECT
      `date`,
      `type` as holiday_type,
      locale_name
    FROM
      holiday_events
    WHERE
      locale == "Local"
    GROUP BY
      ALL
    ORDER BY
      `date`)
""")

sql(
"""
CREATE OR REPLACE TABLE local_holidays_silver AS (
SELECT
  `date`,
  store_nbr,
  CASE
    WHEN num_holidays >= 2 THEN "Multiple"
    ELSE local_holiday_type
  END as local_holiday_type
FROM
  (
    SELECT
      h.`date`,
      s.store_nbr,
      MIN(h.holiday_type) as local_holiday_type,
      count(1) as num_holidays
    FROM
      favorita_stores s
      INNER JOIN local_holidays_bronze h ON (
        s.city == h.locale_name
        OR s.state == h.locale_name
      )
    GROUP BY
      ALL
    ORDER BY
      h.`date`
  )
)
""")
display(sql("SELECT * FROM local_holidays_silver LIMIT 5"))

# COMMAND ----------

# DBTITLE 1,Regional holidays
sql("""
  CREATE OR REPLACE TABLE regional_holidays_bronze AS (
    SELECT
      `date`,
      `type` as holiday_type,
      locale_name
    FROM
      holiday_events
    WHERE
      locale == "Regional"
    GROUP BY
      ALL
    ORDER BY
      `date`)
""")

sql(
"""
CREATE OR REPLACE TABLE regional_holidays_silver AS (
SELECT
  `date`,
  store_nbr,
  CASE
    WHEN num_holidays >= 2 THEN "Multiple"
    ELSE regional_holiday_type
  END as regional_holiday_type
FROM
  (
    SELECT
      h.`date`,
      s.store_nbr,
      MIN(h.holiday_type) as regional_holiday_type,
      count(1) as num_holidays
    FROM
      favorita_stores s
      INNER JOIN regional_holidays_bronze h ON (
        s.city == h.locale_name
        OR s.state == h.locale_name
      )
    GROUP BY
      ALL
    ORDER BY
      h.`date`
  )
)
""")
display(sql("SELECT * FROM regional_holidays_silver LIMIT 5"))

# COMMAND ----------

# DBTITLE 1,National holidays
sql("""
  CREATE OR REPLACE TABLE national_holidays_bronze AS (
    SELECT
      `date`,
      MIN(`type`) as holiday_type,
      count(1) as num_holidays
    FROM
      holiday_events
    WHERE
      locale == "National"
    GROUP BY ALL
    ORDER BY
      `date`)
""")

sql(
"""
CREATE OR REPLACE TABLE national_holidays_silver AS (
SELECT
  `date`,
  store_nbr,
  national_holiday_type
FROM
  (
    SELECT
      `date`,
      CASE
        WHEN num_holidays >= 2 THEN "Multiple"
        ELSE holiday_type
      END as national_holiday_type
    FROM
      national_holidays_bronze h
    GROUP BY
      ALL
    ORDER BY
      h.`date`
  ), favorita_stores 
)
""")
display(sql("SELECT * FROM national_holidays_silver LIMIT 5"))

# COMMAND ----------

# DBTITLE 1,Holiday feature table
df = sql("""
      SELECT
        ifnull(n.`date`,ifnull(r.`date`, l.`date`)) as `date`,
        ifnull(n.store_nbr,ifnull(r.store_nbr, l.store_nbr)) as store_nbr,
        n.national_holiday_type,
        r.regional_holiday_type,
        l.local_holiday_type
      FROM
        national_holidays_silver n
      FULL JOIN regional_holidays_silver r ON n.`date`=r.`date` AND n.store_nbr = r.store_nbr
      FULL JOIN local_holidays_silver l ON n.`date`=l.`date` AND n.store_nbr = l.store_nbr
      ORDER BY
        `date`
      """)

df.limit(5).display()

fe.create_table(
    name = f"{database_name}.store_holidays_ft",
    primary_keys=["date", "store_nbr"],
    df=df,
    description="Holidays in Ecuador by date and store number. Table includes holiday types for national, regional, and local. Nulls indicate non-holiday days.",
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM store_holidays_ft

# COMMAND ----------

# DBTITLE 1,Lagging the oil price for a proxy of the economy
df = sql(
    """
    SELECT
      date(`date`),
      date(`date`) -10 as join_on_date,
      dcoilwtico as lag10_oil_price
    FROM
      oil_prices
  """
)

fe.create_table(
    name=f"{database_name}.oil_10d_lag_ft",
    primary_keys=["join_on_date"],
    df=df,
    description="The lag10_oil_price is the price of oil 10 days after the join_on_date.",
)
