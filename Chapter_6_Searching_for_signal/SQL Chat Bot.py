# Databricks notebook source
# MAGIC %md
# MAGIC # SQL Bot
# MAGIC
# MAGIC We will use an LLM model from OpenAI's ChatGPT to ask questions of our Favorita Sales tables
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Up

# COMMAND ----------

# MAGIC %pip install openai sqlalchemy-databricks

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=sql-bot $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Connect to OpenAI

# COMMAND ----------

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope="dlia", key="OPENAI_API_KEY")
llm = OpenAI(temperature=0)
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Collect the tables of interest and their metadata

# COMMAND ----------

schema = "favorita_forecasting"

table_schemas = spark.sql(
f"""
select
  * 
from
  system.information_schema.columns 
where
  table_catalog = '{catalog}'
  and table_schema = '{schema}'
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
    records = spark.sql(f"select * from {catalog}.{schema}.{table} limit 2")

    return f"Example records for {table}: \ncolumn_index" + records.toPandas().to_csv(
        sep="|"
    )

# COMMAND ----------

metadata = {}
table_list = []

for row in table_schemas.select("table_name").distinct().collect():
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
from langchain import SQLDatabase, SQLDatabaseChain

table = table_list

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
# Obtain this from a SQL endpoint under "Connection Details", HTTP Path
endpoint_http_path = "/sql/1.0/warehouses/475b94ddc7cd5211"

engine = create_engine(
  f"databricks+connector://token:{databricks_token}@{workspace_url}:443/{schema}",
  connect_args={"http_path": endpoint_http_path, "catalog": catalog}
)

db = SQLDatabase(engine, schema=None, include_tables=table) 
# schema=None to work around https://github.com/hwchase17/langchain/issues/2951 ?
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
chat_chain = SQLDatabaseChain.from_llm(chat, db, verbose=True)

# COMMAND ----------

question = "Which store sold the most?"

# COMMAND ----------

db_chain.run(question)

# COMMAND ----------

chat_chain.run(question)
