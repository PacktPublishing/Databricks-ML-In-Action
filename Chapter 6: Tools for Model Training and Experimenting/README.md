# Chapter 6: Searching for Signal

**Here is what you will learn as part of this chapter:**
1. Building a training set from a feature table
2. Baselining with AutoML
3. Tracking experiments with MLflow
4. Classifying beyond the basic
5. Integrating innovation
6. Applying our learning

## Technical requirements 

Here are the technical requirements needed to complete the hands-on examples in this chapter:
- In order to use the OpenAI API, you need to [set up a paid account](https://platform.openai.com/account/billing/overview).
- For our LLM model, we will integrate with the ChatGPT model from OpenAI. You will need an [API Key](https://platform.openai.com/account/api-keys) and install the [OpenAI Python library](https://pypi.org/project/openai/)
- We use the [SQLAlchemy Dialect](https://pypi.org/project/sqlalchemy-databricks/) for Databricks workspace and sql analytics clusters using the officially supported databricks-sql-connector dbapi.
  
## Links

**In the chapter**
- [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html#tracking)
- [MLFlow Model flavors](https://mlflow.org/docs/latest/models.html#built-in-model-flavors)
- [Introducing the Spark PyTorch Distributor](https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html)
- [Data & AI Summit 2023: Generative AI at Scale Using GAN and Stable Diffusion](https://www.youtube.com/watch?v=YsWZDCsM9aE)
- [New Expert-Led Large Language Models (LLMs) Courses on edX](https://www.databricks.com/blog/enroll-our-new-expert-led-large-language-models-llms-courses-edx)
- [OpenAI](https://platform.openai.com)

**Further Reading**
- [Introducing AI Functions: Integrating Large Language Models with Databricks SQL](https://www.databricks.com/blog/2023/04/18/introducing-ai-functions-integrating-large-language-models-databricks-sql.html)
- [Deploy Your LLM Chatbot With Retrieval Augmented Generation (RAG), llama2-70B (MosaicML inferences) and Vector Search](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot)
- [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
- [Unifying Your Data Ecosystem with Delta Lake Integration](https://www.databricks.com/blog/integrating-delta-lakehouse-other-platforms)
- [Reading and Writing from and to Delta Lake from non-Databricks platforms](https://www.databricks.com/blog/integrating-delta-lakehouse-other-platforms)
- [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- [Ray 2.3 release (PyPI)](https://pypi.org/project/ray/)
- [Ray on Spark Databricks docs](https://docs.databricks.com/machine-learning/ray-integration.html)
- [Announcing Ray support on Databricks and Apache Spark Clusters Blog post](https://www.databricks.com/blog/2023/02/28/announcing-ray-support-databricks-and-apache-spark-clusters.html)
- [Ray docs](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/spark.html#deploying-on-spark-standalone-cluster)
- [Databricks Blog: Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
