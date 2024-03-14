# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.12.0 databricks-genai-inference==0.1.1 mlflow==2.9.0 textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 langchain==0.0.344 databricks-vectorsearch==0.22 transformers==4.30.2 torch==2.0.1 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=rag_chatbot

# COMMAND ----------

import mlflow
from mlia_utils.mlflow_funcs import * 
from pyspark.sql.functions import col, udf, length, pandas_udf, explode

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("mlaction", "rag_sp_token")
model_name = f"{catalog}.{database_name}.mlaction_chatbot_model"

model_version_to_evaluate = get_latest_model_version(model_name)
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

# COMMAND ----------

question = "What is GPT ?"
dialog = {"query": question}
rag_model.invoke({"query":"What is GPT? "})["result"]

# COMMAND ----------

@pandas_udf("string")
def predict_answer(questions):
    def answer_question(question):
        dialog = {"query": question}
        return rag_model.invoke(dialog)['result']
    return questions.apply(answer_question)

# COMMAND ----------

df_qa = (spark.read.table('evaluation_table')
                  .selectExpr('question_asked as inputs', 'answer_given as targets')
                  .where("targets is not null")
                  #.sample(fraction=0.005, seed=40) # if your dataset is very big you could sample it 
                  ) 

df_qa_with_preds = df_qa.withColumn('preds', predict_answer(col('inputs'))).cache()
display(df_qa_with_preds)

# COMMAND ----------

df_qa_with_preds.write.mode("overwrite").saveAsTable(f"{catalog}.{database_name}.evaluation_table_preds")

# COMMAND ----------

# MAGIC %md 
# MAGIC You could also use OpenAI GPT4 as your judge! 
# MAGIC Here is the example how to set it with the Azure OpenAI the deployment endpoints: 
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC try:
# MAGIC     endpoint_name  = "mlaction-azure-openai"
# MAGIC     deploy_client.create_endpoint(
# MAGIC         name=endpoint_name,
# MAGIC         config={
# MAGIC             "served_entities": [
# MAGIC                 {
# MAGIC                     "name": endpoint_name,
# MAGIC                     "external_model": {
# MAGIC                         "name": "gpt-35-turbo",
# MAGIC                         "provider": "openai",
# MAGIC                         "task": "llm/v1/chat",
# MAGIC                         "openai_config": {
# MAGIC                             "openai_api_type": "azure",
# MAGIC                             "openai_api_key": "{{secrets/mlaction/azure-openai}}", #Replace with your own azure open ai key
# MAGIC                             "openai_deployment_name": "mlaction-gpt35",
# MAGIC                             "openai_api_base": "https://mlaction-open-ai.openai.azure.com/",
# MAGIC                             "openai_api_version": "2023-05-15"
# MAGIC                         }
# MAGIC                     }
# MAGIC                 }
# MAGIC             ]
# MAGIC         }
# MAGIC     )
# MAGIC except Exception as e:
# MAGIC     if 'RESOURCE_ALREADY_EXISTS' in str(e):
# MAGIC         print('Endpoint already exists')
# MAGIC     else:
# MAGIC         print(f"Couldn't create the external endpoint with Azure OpenAI: {e}. Will fallback to llama2-70-B as judge. Consider using a stronger model as a judge.")
# MAGIC         endpoint_name = "databricks-llama-2-70b-chat"
# MAGIC ```

# COMMAND ----------

from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")
endpoint_name = "databricks-llama-2-70b-chat"
#Let's query our external model endpoint
answer_test = deploy_client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is GPT?"}]})
answer_test['choices'][0]['message']['content'] 

# COMMAND ----------

from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Because we have our labels (answers) within the evaluation dataset, we can evaluate the answer correctness as part of our metric. Again, this is optional.
answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")
print(answer_correctness_metrics)

# COMMAND ----------

# Adding custom professionalism metric
professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

print(professionalism)

# COMMAND ----------

from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

#This will automatically log all
with mlflow.start_run(run_name="chatbot_rag") as run:
    eval_results = mlflow.evaluate(data = df_qa_with_preds.toPandas(), # evaluation data,
                                   model_type="question-answering", # toxicity and token_count will be evaluated   
                                   predictions="preds", # prediction column_name from eval_df
                                   targets = "targets",
                                   extra_metrics=[answer_correctness_metrics, professionalism])
    
eval_results.metrics

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

import plotly.express as px
px.histogram(df_genai_metrics, x="token_count", labels={"token_count": "Token Count"}, title="Distribution of Token Counts in Model Responses")

# COMMAND ----------

# Counting the occurrences of each answer correctness score
px.bar(df_genai_metrics['professionalism/v1/score'].value_counts(), title='Answer Professionalism Score Distribution')

# COMMAND ----------

# Counting the occurrences of each answer professionalism score
px.bar(df_genai_metrics['answer_correctness/v1/score'].value_counts(), title='Answer Correctness Score Distribution')

# COMMAND ----------

df_genai_metrics['toxicity'] = df_genai_metrics['toxicity/v1/score'] * 100
fig = px.scatter(df_genai_metrics, x='toxicity', y='answer_correctness/v1/score', title='Toxicity vs Correctness', size=[10]*len(df_genai_metrics))
fig.update_xaxes(tickformat=".2f")

# COMMAND ----------

# MAGIC %md
# MAGIC  ## This is looking good, let's tag our model as production ready
# MAGIC After reviewing the model correctness and potentially comparing its behavior to your other previous version, we can flag our model as ready to be deployed.
# MAGIC *Note: Evaluation can be automated and part of a MLOps step: once you deploy a new Chatbot version with a new prompt, run the evaluation job and benchmark your model behavior vs the previous version.*
# MAGIC
# MAGIC

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="Production", version=model_version_to_evaluate)

# COMMAND ----------


