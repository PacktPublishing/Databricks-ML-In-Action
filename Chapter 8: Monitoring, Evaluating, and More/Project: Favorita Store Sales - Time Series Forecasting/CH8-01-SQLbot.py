# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 8: Monitoring, Evaluating, and More
# MAGIC
# MAGIC ## Favorita Sales - SQLbot
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Library and Restart Python

# COMMAND ----------

# MAGIC %pip install openai sqlalchemy-databricks langchain_openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=favorita_forecasting

# COMMAND ----------

# MAGIC %md We will use an LLM model from OpenAI's ChatGPT to ask questions of our Favorita Sales tables

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Connect to OpenAI

# COMMAND ----------

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope="dlia", key="OPENAI_API_KEY")
llm = OpenAI(temperature=0)
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Collect the tables of interest and their metadata

# COMMAND ----------

table_schemas = spark.sql(
f"""
select
  * 
from
  system.information_schema.columns 
where
  table_catalog = '{catalog}'
  and table_schema = '{database_name}'
order by
  table_name,
  ordinal_position;
"""
)

table_schemas = table_schemas.drop(
        "table_catalog",
        "table_schema",
        "ordinal_position",
        "character_octet_length",
        "numeric_precision",
        "numeric_precision_radix",
        "numeric_scale",
        "datetime_precision",
        "interval_type",
        "interval_precision",
        "identity_start",
        "identity_increment",
        "identity_maximum",
        "identity_minimum",
        "identity_cycle",
        "is_system_time_period_start",
        "is_system_time_period_end",
        "system_time_period_timestamp_generation",
        "is_updatable"
    )

# COMMAND ----------

def table_def(table):
    table_schema = table_schemas.drop("table_name").where(f'table_name = "{table}"')

    return f"Table Schema for {table}: \ncolumn_index" + table_schema.toPandas().to_csv(
        sep="|"
    )

def table_records(table):
    records = spark.sql(f"select * from {catalog}.{database_name}.{table} limit 2")

    return f"Example records for {table}: \ncolumn_index" + records.toPandas().to_csv(
        sep="|"
    )

# COMMAND ----------

metadata = {}
table_list = []
iter_tables = table_schemas.select("table_name").distinct().collect()

for row in iter_tables:
  tbl = row['table_name']
  table_list.append(tbl)
  meta = table_def(tbl) + '\n\n' + table_records(tbl)
  metadata[tbl] = meta

# print(metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a connection to Databricks SQL Warehouse

# COMMAND ----------

from sqlalchemy.engine import create_engine
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

table = table_list

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
# Obtain this from a SQL endpoint under "Connection Details", HTTP Path
endpoint_http_path = "/sql/1.0/warehouses/5ab5dda58c1ea16b"

engine = create_engine(
  f"databricks+connector://token:{databricks_token}@{workspace_url}:443/{database_name}",
  connect_args={"http_path": endpoint_http_path, "catalog": catalog}
)

db = SQLDatabase(engine, schema=None, include_tables=table)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
chat_chain = SQLDatabaseChain.from_llm(chat, db, verbose=True)

# COMMAND ----------

question = "Which store sold the most?"

# COMMAND ----------

db_chain.run(question)

# COMMAND ----------

chat_chain.run(question)

# COMMAND ----------


