# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 4: Exploring and cleaning toward the silver layer
# MAGIC
# MAGIC ## Retrieval Augmented Generation Chatbot - Creating embeddings
# MAGIC https://arxiv.org/pdf

# COMMAND ----------

# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=rag_chatbot

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create Vector Search 
# MAGIC
# MAGIC To learn more about Databricks Vector Search check this documentation: 
# MAGIC - [AWS](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC - [Azure](https://learn.microsoft.com/en-gb/azure/databricks/generative-ai/vector-search) 

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc_endpoint_name = "ml_action_vs"
vsc = VectorSearchClient()

# SIDE NOTE - experience strange behaviour with the Index re-provisioning 
if vsc_endpoint_name not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
  vsc.create_endpoint(name=vsc_endpoint_name, endpoint_type="STANDARD")
  print(f"Endpoint named {vsc_endpoint_name} is in creation, wait a moment.")
 
print(f"Endpoint named {vsc_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md 
# MAGIC Vector Search supports two index types:
# MAGIC
# MAGIC - **Delta Sync Index** automatically syncs with a source Delta table and incrementally updates as the underlying data in the Delta table changes. 
# MAGIC   > You can choose to have *managed embedding* vectors for you or *manage them yourself*.
# MAGIC - **Direct Vector Access Index** supports direct read and write of embedding vectors and metadata through a REST API or an SDK. 
# MAGIC   > For this index, you manage embedding vectors and index updates yourself.

# COMMAND ----------

display(spark.read.table(f"{catalog}.{database_name}.pdf_documentation_text"))

# COMMAND ----------



# COMMAND ----------

from databricks.sdk import WorkspaceClient
from mlia_utils.rag_funcs import *
#import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{database_name}.pdf_documentation_text"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{database_name}.docs_vsc_idx_cont"

if not index_exists(vsc, vsc_endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {vsc_endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=vsc_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", # TRIGGERED or CONTINUOUS 
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(vsc_endpoint_name, vs_index_fullname).sync()


# COMMAND ----------

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vsc_endpoint_name, vs_index_fullname)

# COMMAND ----------

# Checking the information about the VS 
vsc.get_index(vsc_endpoint_name, vs_index_fullname).describe()

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "The Economic Impacts of Automation Technologies using LLMs"

response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(vsc_endpoint_name, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["pdf_name", "content"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])
pprint(docs)


# COMMAND ----------


