# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 3: Building out our Bronze Layer
# MAGIC
# MAGIC ## Retrieval Augmented Generation Chatbot - Extracting chunks
# MAGIC
# MAGIC https://arxiv.org/pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run setup

# COMMAND ----------

# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" pypdf sentence_transformers langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 "git+https://github.com/mlflow/mlflow.git@gateway-migration" typing_extensions==4.7.1
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown(name='Reset', defaultValue='True', choices=['True', 'False'], label="Reset: Drop previous table")

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=rag_chatbot

# COMMAND ----------

documents_folder =  f"{volume_file_path}/raw_documents/"
display(dbutils.fs.ls(documents_folder))

table_name = "pdf_raw"

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating Table with Raw Data 
# MAGIC
# MAGIC This step is optional, you can keep your data in memory if you start with a small volume of examples and not required to keep original files. 

# COMMAND ----------

if bool(dbutils.widgets.get('Reset')):
  sql(f"DROP TABLE IF EXISTS {table_name}")

# COMMAND ----------

df = (
        spark.read.format("BINARYFILE")
        .option("recursiveFileLookup", "true")
        .load('dbfs:'+ documents_folder)
        )

df.write.saveAsTable(f"{catalog}.{database_name}.{table_name}")

# COMMAND ----------

display(sql(f"SELECT * FROM {table_name} LIMIT 2"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Extract Text form PDFs into Chunks

# COMMAND ----------

# DBTITLE 1,Basic extracting
from unstructured.partition.auto import partition
import io

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------

# DBTITLE 1,Quick test
with open(f"{documents_folder}/2303.04671.pdf", mode="rb") as pdf:
  doc = extract_doc_text(pdf.read())  
  print(doc)

# COMMAND ----------


