# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 2: Designing Databricks Day One
# MAGIC
# MAGIC ## Retrieval Augmented Generation Chatbot - Downloading pdf documents
# MAGIC
# MAGIC https://arxiv.org/pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ###Run setup

# COMMAND ----------

dbutils.widgets.dropdown(name='Reset', defaultValue='True', choices=['True', 'False'], label="Reset: Delete previous data")

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=rag_chatbot

# COMMAND ----------

# DBTITLE 1,Notebook Variables
library_folder = "{}/raw_documents".format(volume_file_path)

user_agent = "mlaction_book"

# COMMAND ----------

if bool(dbutils.widgets.get('Reset')):
  dbutils.fs.rm(library_folder, recurse=True)
  dbutils.fs.mkdirs(library_folder)

# COMMAND ----------

# DBTITLE 1,Helper Functions
import os
import requests

def load_file(file_uri, file_name, library_folder):
    
    # Create the local file path for saving the PDF
    local_file_path = os.path.join(library_folder, file_name)

    # Download the PDF using requests
    try:
        # Set the custom User-Agent header
        headers = {"User-Agent": user_agent}

        response = requests.get(file_uri, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the PDF to the local file
            with open(local_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print("PDF downloaded successfully.")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
    except requests.RequestException as e:
        print("Error occurred during the request:", e)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Download documents

# COMMAND ----------

# we are getting our documents, you could directly upload it to the volumes using UI
pdfs = {
        '2312.14565.pdf': 'https://arxiv.org/pdf/2312.14565.pdf', #used for evaluator, created 20 questions from it and answers generated using Mixtral
        '2303.10130.pdf':'https://arxiv.org/pdf/2303.10130.pdf', #used for evaluator, created 20 questions from it and answers generated using Mixtral
        '2302.06476.pdf':'https://arxiv.org/pdf/2312.00506.pdf', 
        '2302.06476.pdf':'https://arxiv.org/pdf/2302.06476.pdf', 
        '2311.07071.pdf':'https://arxiv.org/pdf/2311.07071.pdf',
        '2304.07683.pdf':'https://arxiv.org/pdf/2304.07683.pdf'}

for pdf in pdfs.keys():
    load_file(pdfs[pdf], pdf, library_folder)

# COMMAND ----------

dbutils.fs.ls(library_folder)
