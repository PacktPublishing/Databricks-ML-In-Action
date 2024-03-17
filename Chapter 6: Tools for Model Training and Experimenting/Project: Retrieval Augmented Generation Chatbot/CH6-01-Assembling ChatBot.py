# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 databricks-sdk==0.12.0 mlflow==2.9
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=rag_chatbot

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ###  This Notebook requires a secret to work:
# MAGIC Your Model Serving Endpoint needs a secret to authenticate against your Vector Search Index (see [Documentation](https://docs.databricks.com/en/security/secrets/secrets.html)).  <br/>
# MAGIC **Note: if you are using a shared demo workspace and you see that the secret is setup, please don't run these steps and do not override its value**<br/>
# MAGIC
# MAGIC - You'll need to [setup the Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/install.html) on your laptop or using this cluster terminal: <br/>
# MAGIC `pip install databricks-cli` <br/>
# MAGIC - Configure the CLI. You'll need your workspace URL and a PAT token from your profile page<br>
# MAGIC `databricks configure`
# MAGIC - Create the dbdemos scope:<br/>
# MAGIC `databricks secrets create-scope mlaction`
# MAGIC - Save your service principal secret. It will be used by the Model Endpoint to autenticate. If this is a demo/test, you can use one of your [PAT token](https://docs.databricks.com/en/dev-tools/auth/pat.html).<br>
# MAGIC `databricks secrets put-secret mlaction rag_sp_token`
# MAGIC
# MAGIC *Note: Make sure your service principal has access to the Vector Search index:*
# MAGIC
# MAGIC ```
# MAGIC spark.sql('GRANT USAGE ON CATALOG <catalog> TO `<YOUR_SP>`');
# MAGIC spark.sql('GRANT USAGE ON DATABASE <catalog>.<db> TO `<YOUR_SP>`');
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import databricks.sdk.service.catalog as c
# MAGIC WorkspaceClient().grants.update(c.SecurableType.TABLE, <index_name>, 
# MAGIC                             changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="<YOUR_SP>")])
# MAGIC ```

# COMMAND ----------

from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")
endpoints = deploy_client.list_endpoints()
for endpoint in endpoints:
    print(endpoint['name'])

# COMMAND ----------

import os 
# url used to send the request to your model from the serverless endpoint
#host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
#db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get() 
db_token = dbutils.secrets.get("mlaction", "rag_sp_token")
db_host = dbutils.secrets.get("mlaction", "rag_sp_host")
os.environ['DATABRICKS_TOKEN'] = db_token
os.environ['DATABRICKS_HOST'] = db_host

# COMMAND ----------



# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

# NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
vsc_endpoint_name = "one-env-shared-endpoint-1" #"ml_action_vs"
index_name = f"{catalog}.{database_name}.docs_vsc_idx_cont"

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ['DATABRICKS_HOST'], personal_access_token=os.environ["DATABRICKS_TOKEN"])
    print("\n")
    vs_index = vsc.get_index(
        endpoint_name=vsc_endpoint_name,
        index_name=index_name)
    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model)
    return vectorstore.as_retriever()

# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("What is GPT? ")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

import langchain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

# Using Foundational Model from Databricks Thoughput 
chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", # You could also use Llama70B or GPT4
                            max_tokens = 200
                            )

TEMPLATE = """
You are an assistant for the AI Swat Team. You are answering questions related to the GenerativeAI and LLM's 
and how they impact humans life, labour, economic and financial impact. If the question is not related to one 
of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try 
to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

langchain.debug = False #uncomment to see the chain details and the full prompt being sent
question = {"query": "Will AI impact work forces in the US ? "}
answer = chain.run(question)
print(answer,"\n")

question = {"query": "Can LLM's impact wages and how ? "}
answer = chain.run(question)
print(answer,"\n")

# COMMAND ----------



# COMMAND ----------

import mlflow
from mlia_utils.mlflow_funcs import *
# create experiment if does not exist 
experiment_path = f"/Users/{current_user}/mlaction_chatbot_rag"
mlflow_set_experiment(experiment_path) 

# COMMAND ----------

from mlflow.models import infer_signature
from mlflow.tracking.client import MlflowClient

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{database_name}.mlaction_chatbot_model"

with mlflow.start_run(run_name="mlaction_chatbot_rag") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )
    #------------------------
    import mlflow.models.utils
    mlflow.models.utils.add_libraries_to_model(
        f"models:/{model_name}/{get_latest_model_version(model_name)}"
    )

# COMMAND ----------


