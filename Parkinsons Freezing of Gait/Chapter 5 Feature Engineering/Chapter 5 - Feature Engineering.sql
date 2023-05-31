-- Databricks notebook source
-- MAGIC %run ./../setup

-- COMMAND ----------

USE catalog hive_metastore;
USE lakehouse_in_action;

CREATE OR REPLACE TABLE parkinsons_tiny_fe_n_metadata_w_time AS (SELECT *, row_number() OVER (PARTITION BY `Id` ORDER BY Init) TimeStep FROM parkinsons_tiny_fe_n_metadata)

-- COMMAND ----------


