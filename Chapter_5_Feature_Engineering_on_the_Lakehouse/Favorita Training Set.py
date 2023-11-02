# Databricks notebook source
# MAGIC %md
# MAGIC ## Favorita Forecasting - Build a Training Set
# MAGIC [Kaggle Competition Link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=favorita_forecasting $catalog=lakehouse_in_action

# COMMAND ----------

# DBTITLE 1,Import & Initialize the Feature Store
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### words

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   training_set

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES

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
      INNER JOIN holiday_events h ON (
        s.city == h.locale_name
        OR s.state == h.locale_name
      )
"""
)

display(df)
df = df.drop("type")
fe.create_table(
    f"{database_name}.stores_local_regional_holidays",
    primary_keys=["date", "store_nbr","locale","holiday_type"],
    df=df,
    description="Joined stores with Holidays by locale_name = city or locale_name = state. National holidays are not included.",
)

# COMMAND ----------

# DBTITLE 1,Some dates have more than one national holiday
# MAGIC %sql
# MAGIC SELECT * FROM holiday_events WHERE `date`==date('2016-05-07T00:00:00.000+0000')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE national_holidays AS (
# MAGIC   SELECT
# MAGIC     `date`,
# MAGIC     MIN(`type`) as holiday_type
# MAGIC   FROM
# MAGIC     holiday_events
# MAGIC   WHERE
# MAGIC     locale == "National"
# MAGIC   GROUP BY
# MAGIC     ALL
# MAGIC   ORDER BY
# MAGIC     `date`
# MAGIC );
# MAGIC
# MAGIC SELECT * FROM national_holidays LIMIT 6;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   t.`date`,
# MAGIC   t.store_nbr,
# MAGIC   t.family,
# MAGIC   t.onpromotion,
# MAGIC   count(1) as cnt
# MAGIC FROM
# MAGIC   train_set t
# MAGIC LEFT JOIN national_holidays h ON (date(t.`date`) == date(h.`date`))
# MAGIC GROUP BY ALL
# MAGIC ORDER BY
# MAGIC   cnt DESC

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
        train_set t
        LEFT JOIN national_holidays h ON (date(t.`date`) == date(h.`date`))
      ORDER BY
        t.`date`
      """)

fe.create_table(
    f"{database_name}.train_national_holidays",
    primary_keys=["date", "store_nbr","family","onpromotion"],
    df=df,
    description="Joined the training dataset with National holidays only.",
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT national_holiday_type, count(*) FROM train_national_holidays GROUP BY national_holiday_type

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
    f"{database_name}.oil_lag",
    primary_keys=["join_on_date"],
    df=df,
    description="The lag10_oil_price is the price of oil 10 days after the join_on_date.",
)

# COMMAND ----------

# DBTITLE 1,Update the feature table with a new column
feature_df = fe.read_table(name="train_national_holidays")
oil_df = fe.read_table(name="oil_lag")
oil_df = oil_df.drop("date").withColumnRenamed("join_on_date","date")
feature_df = feature_df.join(oil_df, on=["date"],how="left")

display(feature_df)

# COMMAND ----------

# DBTITLE 1,Overwrite the table to produce a new version
fe.write_table(name="train_national_holidays",
               df=feature_df,
               mode="overwrite")

# COMMAND ----------

# DBTITLE 1,View the version history
# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_national_holidays

# COMMAND ----------

# DBTITLE 1,Select the data from our table as it was for version 5. 
# MAGIC %sql
# MAGIC SELECT * FROM train_national_holidays VERSION AS OF 5 LIMIT 50

# COMMAND ----------

# DBTITLE 1,Select the data from our table as it was for timestamp "2023-10-11 20:41:16.000" 
# MAGIC %sql
# MAGIC SELECT * FROM train_national_holidays TIMESTAMP AS OF "2023-10-11 20:41:16.000" LIMIT 50
