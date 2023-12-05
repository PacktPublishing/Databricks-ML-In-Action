# Building Out Our Bronze Layer 

**Here is what you will learn as part of this chapter:**

1. Revising the Medallion architecture pattern
2. Transforming data to Delta with Auto Loaders
3. Schema Evolution and Delta Live Tables
4. Applying our learning
- Technical Requirements
- Working on Projects

## Links in the chapter
- [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [Delta Live Tables](https://docs.databricks.com/en/delta-live-tables/index.html)
- [Advanced Cluster Options: Spark configuration](https://docs.databricks.com/en/clusters/configure.html#spark-configuration)
- [Spark configurations in the cluster's advanced options](https://docs.databricks.com/en/clusters/configure.html#spark-configuration)

## Technical requirements 
- Databricks ML Runtime includes several pre-installed libraries useful for machine learning and data science projects. For this reason, we will be using clusters with an [ML Runtime](https://docs.databricks.com/runtime/mlruntime.html#introduction-to-databricks-runtime-for-machine-learning).
- [Pandas](https://pandas.pydata.org/)
- [Pandas API on Spark](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)
- [Delta Live Tables (DLT)](https://docs.databricks.com/en/delta-live-tables/index.html)
- [Volumes](https://docs.databricks.com/en/sql/language-manual/sql-ref-volumes.html) 

## Further Reading
-[DLT Databricks Demo](https://www.databricks.com/resources/demos/tutorials/lakehouse-platform/full-delta-live-table-pipeline))
- [AutoLoader options](https://docs.databricks.com/ingestion/auto-loader/options.html)
- [Schema evolution with Auto Loader](https://docs.databricks.com/ingestion/auto-loader/schema.html#configure-schema-inference-and-evolution-in-auto-loader)
- [Common loading patterns with Auto Loader](https://docs.databricks.com/ingestion/auto-loader/patterns.html)
- [Stream processing with Apache Kafka and Databricks](https://docs.databricks.com/structured-streaming/kafka.html)
- [How We Performed ETL on One Billion Records For Under $1 With Delta Live Tables](https://www.databricks.com/blog/2023/04/14/how-we-performed-etl-one-billion-records-under-1-delta-live-tables.html)
- [Create tables - Managed vs External](https://docs.databricks.com/en/data-governance/unity-catalog/create-tables.html#create-tables)
- [Take full advantage of the auto-tuning available](https://docs.databricks.com/delta/tune-file-size.html#configure-delta-lake-to-control-data-file-size)
- [Import Python modules from Databricks repos](https://docs.databricks.com/en/delta-live-tables/import-workspace-files.html)
