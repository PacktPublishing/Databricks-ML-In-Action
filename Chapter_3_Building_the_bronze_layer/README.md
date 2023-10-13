# Building Out Our Bronze Layer 

**Here is what you will learn as part of this chapter:**

1. Revising the Medallion architecture pattern
2. Transforming data to Delta with Auto Loaders
3. Schema Evolution and Delta Live Tables

## Technical requirements 

**Here are the technical requirements needed to complete the hands-on examples in this chapter:**
1. Databricks ML Runtime includes several pre-installed libraries useful for machine learning and data science projects. For this reason, we will be using clusters with an [ML Runtime](https://docs.databricks.com/runtime/mlruntime.html#introduction-to-databricks-runtime-for-machine-learning).
2. [Pandas](https://www.kaggle.com/docs/api](https://pandas.pydata.org/)
3. [Pandas API on Spark](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)
4. [Delta Live Tables (DLT)](https://docs.databricks.com/en/delta-live-tables/index.html)
5. [Volumes](https://docs.databricks.com/en/sql/language-manual/sql-ref-volumes.html) 

## Links

**In the chapter:**
1. [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
2. [Delta Live Tables]([https://docs.databricks.com/data-governance/unity-catalog/index.html](https://docs.databricks.com/en/delta-live-tables/index.html)

**Further Reading:**
- [AutoLoader options](https://docs.databricks.com/ingestion/auto-loader/options.html)
- [Schema evolution with Auto Loader](https://docs.databricks.com/ingestion/auto-loader/schema.html#configure-schema-inference-and-evolution-in-auto-loader)
- [Common loading patterns with Auto Loader](https://docs.databricks.com/ingestion/auto-loader/patterns.html)
- [Stream processing with Apache Kafka and Databricks](https://docs.databricks.com/structured-streaming/kafka.html)
- [How We Performed ETL on One Billion Records For Under $1 With Delta Live Tables](https://www.databricks.com/blog/2023/04/14/how-we-performed-etl-one-billion-records-under-1-delta-live-tables.html)
- [Create tables - Managed vs External](https://docs.databricks.com/en/data-governance/unity-catalog/create-tables.html#create-tables)
- [Take full advantage of the auto-tuning available](https://docs.databricks.com/delta/tune-file-size.html#configure-delta-lake-to-control-data-file-size)
