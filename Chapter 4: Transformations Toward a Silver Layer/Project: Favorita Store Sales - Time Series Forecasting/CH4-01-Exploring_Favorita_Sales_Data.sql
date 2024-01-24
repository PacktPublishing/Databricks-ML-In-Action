-- Databricks notebook source
-- MAGIC %md
-- MAGIC Chapter 4: Exploring and Cleaning Toward a Silver Layer
-- MAGIC
-- MAGIC ## Favorita Sales - Exploring Favorita Sales Data
-- MAGIC
-- MAGIC [Kaggle link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) and [AutoML documentation](https://docs.databricks.com/en/machine-learning/automl/train-ml-model-automl-api.html)

-- COMMAND ----------

-- MAGIC %md ## Run Setup

-- COMMAND ----------

-- MAGIC %run ../../global-setup $project_name=favorita_forecasting

-- COMMAND ----------

SELECT * FROM train_set

-- COMMAND ----------

SELECT ts.* FROM train_set ts 
INNER JOIN (SELECT family, sum(sales) as total_sales FROM train_set GROUP BY family ORDER BY total_sales DESC LIMIT 10) top_10 ON ts.family=top_10.family

-- COMMAND ----------

SELECT COUNT(DISTINCT store_nbr) AS num_stores, COUNT(DISTINCT type) AS num_store_types
FROM favorita_stores
WHERE state='Guayas'

-- COMMAND ----------

SELECT type, COUNT(*) AS num_stores
FROM favorita_stores
WHERE state='Guayas'
GROUP BY all
ORDER BY num_stores DESC

-- COMMAND ----------

SELECT type, COUNT(*) AS num_stores
FROM favorita_stores
WHERE state = 'Guayas'
GROUP BY type
ORDER BY num_stores DESC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from databricks import automl
-- MAGIC summary = automl.regress(df, target_col="sales", timeout_minutes=30)

-- COMMAND ----------


