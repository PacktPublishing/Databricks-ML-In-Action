# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Searching for Signal
# MAGIC
# MAGIC ##Synthetic data - Creating a labeled dataset

# COMMAND ----------

# MAGIC %pip install dbldatagen

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=synthetic_transactions $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate data using DB labs data generator
# MAGIC When combining dataframes, UNION returns distinct records from both of the tables, while UNION ALL returns all the records from both the tables (UNION ALL is faster)

# COMMAND ----------

data_rows = 1000
label_rows = .3 * data_rows

# COMMAND ----------

import dbldatagen as dg
import dbldatagen.distributions as dist
from pyspark.sql.types import IntegerType, FloatType, StringType

dfa_spec0 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=data_rows, partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct \A")
           .withColumn("Amount", FloatType(), minValue=1000,maxValue=25001, 
                       distribution=dist.Normal(mean=15520, stddev=.001), random=True)
           .withColumn("Label", IntegerType(), minValue=0, maxValue=0)
           )

dfa_spec1 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=label_rows, partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct \A")
           .withColumn("Amount", FloatType(), minValue=1000,maxValue=25001, 
                       distribution=dist.Beta(alpha=4, beta=10), random=True)
           .withColumn("Label", IntegerType(), minValue=1, maxValue=1)
           )

dfa0 = dfa_spec0.build()
dfa1 = dfa_spec1.build()
df = dfa1.unionAll(dfa0)

# COMMAND ----------

dfb_spec0 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=data_rows,partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct B")
           .withColumn("Amount", FloatType(), minValue=1000,maxValue=55001, 
                       distribution=dist.Normal(mean=35520, stddev=.001), random=True)
           .withColumn("Label", IntegerType(), minValue=0, maxValue=0)
           )

dfb_spec1 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=label_rows,partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct B")
           .withColumn("Amount", FloatType(), minValue=1000,maxValue=55001, 
                       distribution=dist.Beta(alpha=10, beta=4), random=True)
           .withColumn("Label", IntegerType(), minValue=1, maxValue=1))           

df = df.unionAll(dfb_spec0.build()).unionAll(dfb_spec1.build())                          

dfc_spec0 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=data_rows,partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct \C")
           .withColumn("Amount", FloatType(), minValue=1000,maxValue=40001, 
                       distribution=dist.Normal(mean=30520, stddev=.001), random=True)
           .withColumn("Label", IntegerType(), minValue=0, maxValue=0)
           )

dfc_spec1 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=data_rows, partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct \C")
           .withColumn("Amount", FloatType(), minValue=1000,maxValue=40001,  
                       distribution=dist.Beta(alpha=3, beta=10), random=True)
           .withColumn("Label", IntegerType(), minValue=1, maxValue=1))

df = df.unionAll(dfc_spec0.build()).unionAll(dfc_spec1.build())

dfd_spec0 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=data_rows, partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct \D")
           .withColumn("Amount", FloatType(), minValue=100,maxValue=40001, 
                       distribution=dist.Normal(mean=5200, stddev=.001), random=True)
           .withColumn("Label", IntegerType(), minValue=0, maxValue=0)
           )
dfd_spec1 = (dg.DataGenerator(spark, name="synthetic_transactions", rows=data_rows, partitions=4)
           .withColumn("CustomerID", IntegerType(), minValue=1234, maxValue=1260, random=True)
           .withColumn("TransactionTimestamp", "timestamp", 
                       begin="2024-01-01 01:00:00", end="2024-01-31 23:59:00",
                      interval="15 seconds", random=True)
           .withColumn("Product", StringType(), template="Pro\duct \D")
           .withColumn("Amount", FloatType(), minValue=100,maxValue=40001,  
                       distribution=dist.Beta(alpha=8, beta=2), random=True)
           .withColumn("Label", IntegerType(), minValue=1, maxValue=1))

df = df.unionAll(dfd_spec0.build()).unionAll(dfd_spec1.build())

# COMMAND ----------

from pyspark.sql.functions import expr
df = df.withColumn("Amount", expr("Amount / 100"))
df.orderBy("TransactionTimestamp").write.mode("overwrite").saveAsTable("labeled_transactions")
display(df)
