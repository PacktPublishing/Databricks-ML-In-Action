# Chapter 2: Designing Databricks Day One

**Here is what you will learn as part of this chapter:**
1. Planning your Lakehouse 
2. Applying our learning
 
## Technical requirements 

Here are the technical requirements needed to complete the hands-on examples in this chapter:
- We utilize a Python package, opendatasets, to download the data we need from the Kaggle API easily. 
- We use the Databricks Labs Python library, dbldatagen, to generate synthetic data.  
- To use the Kaggle API, you must download your credential file, kaggle.json.  
- A GitHub account is beneficial for connecting Databricks and the code repository for the book (https://github.com/PacktPublishing/Databricks-Lakehouse-ML-In-Action). In addition to a GitHub account, it is ideal to fork the book repository into your GitHub account. You will see that each chapter has a folder, and each project has a folder under the chapters. We will refer to the notebooks by name throughout the project work. 
- We will use the Databricks Secrets API to save both Kaggle and OpenAI credentials. The Secrets API requires the Databricks CLI.  We will walk through this setup. However, you will need to create a personal access token (PAT) on your own for the configuration step. https://docs.databricks.com/en/dev-tools/auth/pat.html 


## Links 

**In the chapter**

**Further Reading**
- [What is Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [Lakehouse Monitoring demo](https://youtu.be/3TLBZSKeYTk?t=560)
- [UC has a more centralized method of managing the model lifecycle than HMS](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
- [Share Models across workspaces](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/multiple-workspaces.html)
- [In depth UC setup on Azure](https://youtu.be/itGKRVHdNPo)
- [Share models across workspaces](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/multiple-workspaces.html)
- [UC Limitations](https://docs.databricks.com/en/data-governance/unity-catalog/index.html#unity-catalog-limitations)
- [Best practices: Cluster configuration | Select Cloud in the dropdown](https://docs.databricks.com/clusters/cluster-config-best-practices.html)
- [Best practices for DBFS and Unity Catalog | Select Cloud in the dropdown](https://docs.databricks.com/dbfs/unity-catalog.html)
- [Databricks Notebooks](https://docs.databricks.com/en/notebooks/index.html)
- [Databricks Autologging | Select Cloud in the dropdown](https://docs.databricks.com/mlflow/databricks-autologging.html#security-and-data-management)
- [Kaggle API GitHub](https://github.com/Kaggle/kaggle-api)
- [Lakehouse Monitoring: Intelligent data and model monitoring](https://www.databricks.com/product/machine-learning/lakehouse-monitoring)
- [System Tables: Billing Forecast, Usage Analytics, and Access Auditing With Databricks Unity Catalog](https://www.databricks.com/resources/demos/tutorials/governance/system-tables)
- [Opendatasets python package](https://pypi.org/project/opendatasets/)
- [Kaggle API](https://www.kaggle.com/docs/api)
- [GitHub](https://github.com/)
- [GitHub: About personal access tokense](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#about-personal-access-tokens)
- [Set up Databricks Repose](https://docs.databricks.com/en/repos/repos-setup.html)
- [Databricks ML in Action Github Repositorye](https://github.com/PacktPublishing/Databricks-Lakehouse-ML-In-Action) 
- [Databricks Secrets APIe](https://docs.databricks.com/en/security/secrets/secrets.html)
- [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html)
- [Databricks Utilities](https://docs.databricks.com/en/dev-tools/databricks-utils.html)
- [Workspace libraries](https://docs.databricks.com/en/libraries/workspace-libraries.html)
- [Data Mesh and the DI Platforms Blog Posts 1](https://www.databricks.com/blog/2022/10/10/databricks-lakehouse-and-data-mesh-part-1.html)
- [Data Mesh and the DI Platforms Blog Posts 2](https://www.databricks.com/blog/2022/10/19/building-data-mesh-based-databricks-lakehouse-part-2.html)
- [Short YouTube video on managed vs external tables in UC](https://youtu.be/yt9vax_PH58?si=dVJRZHAOnrEUBdkA)
- [Query Federation](https://docs.databricks.com/en/query-federation/index.html)
- [Centralized model registry workspace for HMS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/multiple-workspaces.html)
- [Manage model lifecycle in Unity Catalog](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
- [Terraform](https://github.com/databricks/terraform-provider-databricks)
- [Widgets](https://docs.databricks.com/notebooks/widgets.html)
- [Kaggle API GitHub](https://github.com/Kaggle/kaggle-api)




