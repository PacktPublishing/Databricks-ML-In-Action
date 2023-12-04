# Chapter 2: Designing Databricks Day One

**Here is what you will learn as part of this chapter:**

1. Planning your Lakehouse 
2. Leading with business value 
3. Applying our learning
- Technical Requirements
- Setting up your workspace
- Starting the projects

## Links in the chapter

- [Run queries using Lakehouse Federation](https://docs.databricks.com/en/query-federation/index.html)
- [What is Unity Catalog?](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [Share models across workspaces](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/multiple-workspaces.html)
- [UC Limitations](https://docs.databricks.com/data-governance/unity-catalog/index.html#general-limitations)
- [Using the UC Model Registry](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
- [Extending Databricks Unity Catalog with an Open Apache Hive Metastore API](https://www.databricks.com/blog/extending-databricks-unity-catalog-open-apache-hive-metastore-api)
- [Lakehouse Monitoring: Intelligent data and model monitoring](https://www.databricks.com/product/machine-learning/lakehouse-monitoring)
- [Categorize customer requests into request types](https://support.atlassian.com/jira-service-management-cloud/docs/categorize-customer-requests-into-request-types/)
- [System Tables: Billing Forecast, Usage Analytics, and Access Auditing With Databricks Unity Catalog](https://www.databricks.com/resources/demos/tutorials/governance/system-tables)
- [Azure Marketplace](https://azure.microsoft.com/en-us/products/databricks), [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-wtyi5lgtce6n6), [Try Databricks](https://www.databricks.com/try-databricks)


## Technical requirements

- We utilize a Python package, opendatasets, to download the data we need from the Kaggle API with ease: [opendatasets](https://pypi.org/project/opendatasets/)
- To use the Kaggle API, you must download your credential file, kaggle.json. [Kaggle API](https://www.kaggle.com/docs/api)
- A GitHub account is very helpful for connecting Databricks and the code repository for the book: [GitHub](https://github.com/)
- In addition to a GitHub account, it is ideal to fork the book repository into your GitHub account: [Databricks Lakehouse ML in Action repository](https://github.com/PacktPublishing/Databricks-Lakehouse-ML-In-Action)
- [Databricks Secrets API](https://docs.databricks.com/en/security/secrets/secrets.html)
- [What is the Databricks CLI?](https://docs.databricks.com/en/dev-tools/cli/index.html)
- [Install or update the Databricks CLI]( https://docs.databricks.com/en/dev-tools/cli/install.html)
- You will need to download the Tensorlite wheel from Kaggle to get the specific version used in competition: [Tensorlite wheel](https://www.kaggle.com/datasets/philculliton/tflite-wheels-2140)

## Further Reading
- [Lakehouse Monitoring demo](https://youtu.be/3TLBZSKeYTk?t=560)
- [UC has a more centralized method of managing the model lifecycle than HMS](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
- [In depth UC setup on Azure](https://youtu.be/itGKRVHdNPo)
- [Best practices: Cluster configuration | Select Cloud in the dropdown](https://docs.databricks.com/clusters/cluster-config-best-practices.html)
- [Best practices for DBFS and Unity Catalog | Select Cloud in the dropdown](https://docs.databricks.com/dbfs/unity-catalog.html)
- [Databricks Notebooks](https://docs.databricks.com/en/notebooks/index.html)
- [Databricks Autologging | Select Cloud in the dropdown](https://docs.databricks.com/mlflow/databricks-autologging.html#security-and-data-management)
- [Kaggle API GitHub](https://github.com/Kaggle/kaggle-api)
- [Jira tickets](https://support.atlassian.com/jira-service-management-cloud/docs/categorize-customer-requests-into-request-types/)
- [Databricks Utilities](https://docs.databricks.com/en/dev-tools/databricks-utils.html)
- [Workspace libraries](https://docs.databricks.com/en/libraries/workspace-libraries.html)




