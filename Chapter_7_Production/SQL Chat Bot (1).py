# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Production
# MAGIC
# MAGIC ## Favorita Sales - SQL Chat Bot (1)
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Library and Restart Python

# COMMAND ----------

# MAGIC %pip install openai sqlalchemy-databricks

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=sql-bot $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md We will use an LLM model from OpenAI's ChatGPT to ask questions of our Favorita Sales tables

# COMMAND ----------

import sys
import os

sys.path.append(os.path.abspath('/Repos/stephanie.rivera@databricks.com/Databricks-Lakehouse-ML-in-Action/Chapter_6_Searching_for_signal'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Connect to OpenAI

# COMMAND ----------

# MAGIC %run ./OpenAI_setup

# COMMAND ----------

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0)
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Collect the tables of interest and their metadata

# COMMAND ----------

schema = "favorita_forecasting"

# COMMAND ----------

table_schemas = spark.sql(
    f"""
    with constraints as (
      select
        k.*,
        cs.constraint_type,
        u.table_catalog referential_table_catalog,
        u.table_schema referential_table_schema,
        u.table_name referential_table_name
      from
        system.information_schema.key_column_usage k
        inner join system.information_schema.table_constraints cs on k.constraint_catalog = cs.constraint_catalog
        and k.constraint_schema = cs.constraint_schema
        and k.constraint_name = cs.constraint_name
        left outer join (
          select
            distinct constraint_catalog,
            constraint_schema,
            constraint_name,
            table_catalog,
            table_schema,
            table_name
          from
            system.information_schema.constraint_column_usage
        ) u on k.constraint_catalog = u.constraint_catalog
        and k.constraint_schema = u.constraint_schema
        and k.constraint_name = u.constraint_name
        and cs.constraint_type = 'FOREIGN KEY'
      where
        k.table_catalog = '{catalog}'
        and k.table_schema = '{schema}'
    )
    select
  c.*,
  cs.constraint_name,
  cs.ordinal_position constraint_ordinal_position,
  cs.constraint_type,
  cs.referential_table_catalog,
  cs.referential_table_schema,
  cs.referential_table_name
from
  system.information_schema.columns c
  left outer join constraints cs on c.table_catalog = cs.table_catalog
  and c.table_schema = cs.table_schema
  and c.table_name = cs.table_name
  and c.column_name = cs.column_name
where
  c.table_catalog = '{catalog}'
  and c.table_schema = '{schema}'
order by
  table_name,
  ordinal_position;
  """
)


def table_def(table):
    table_schema = table_schemas.drop(
        "table_catalog",
        "table_schema",
        "table_name",
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
        "is_updatable",
    ).where(f'table_name = "{table}"')

    return f"Table Schema for {table}: \ncolumn_index" + table_schema.toPandas().to_csv(
        sep="|"
    )


def table_records(table):
    records = spark.sql(f"select * from {catalog}.{schema}.{table} limit 2")

    return f"Example records for {table}: \ncolumn_index" + records.toPandas().to_csv(
        sep="|"
    )

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   system.information_schema.tables t
# MAGIC where
# MAGIC   t.table_catalog = "lakehouse_in_action"
# MAGIC   and t.table_schema = "favorita_forecasting"

# COMMAND ----------

metadata = {}

sql =f"""
  select
      *
    from
      system.information_schema.tables t
    where
      t.table_catalog = "{catalog}"
      and t.table_schema = "{schema}"
    """

table_list = []
for row in spark.sql(sql).collect():
  tbl = row['table_name']
  table_list.append(tbl)
  meta = table_def(tbl) + '\n\n' + table_records(row['table_name'])
  metadata[tbl] = meta

#print(metadata)

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

db = SQLDatabase(engine, schema=None, include_tables=table) # schema=None to work around https://github.com/hwchase17/langchain/issues/2951 ?
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
chat_chain = SQLDatabaseChain.from_llm(chat, db, verbose=True)

# COMMAND ----------

question = "Which store sold the most?"

# COMMAND ----------

db_chain.run(question)

# COMMAND ----------

chat_chain.run(question)
