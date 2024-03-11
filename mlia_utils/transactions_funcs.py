import dbldatagen as dg
from datetime import datetime
import dbldatagen.distributions as dist
from pyspark.sql.types import IntegerType, FloatType, StringType

def define_specs(context,productName, pVars, Label,CustomerID_vars, currentTimestamp = datetime.now()):
  if Label:
    return (dg.DataGenerator(context, name="syn_trans1", rows=1, partitions=4)
      .withColumn("CustomerID", IntegerType(), nullable=False,
                  minValue=CustomerID_vars["min"], maxValue=CustomerID_vars["max"], random=True)
      .withColumn("TransactionTimestamp", "timestamp", 
                  begin=currentTimestamp, end=currentTimestamp,nullable=False,
                random=False)
      .withColumn("Product", StringType(), template=f"Pro\duct \{productName}") 
      .withColumn("Amount", FloatType(), 
                  minValue=pVars["min"],maxValue=pVars["max"], 
                  distribution=dist.Beta(alpha=pVars["alpha"], beta=pVars["beta"]), random=True)
      .withColumn("Label", IntegerType(), minValue=1, maxValue=1)).build()
  else:
    return (dg.DataGenerator(context, name="syn_trans0", rows=1, partitions=4)
      .withColumn("CustomerID", IntegerType(), nullable=False,
                  minValue=CustomerID_vars["min"], maxValue=CustomerID_vars["max"], random=True)
      .withColumn("TransactionTimestamp", "timestamp", 
                  begin=currentTimestamp, end=currentTimestamp,nullable=False,
                random=False)
      .withColumn("Product", StringType(), template=f"Pro\duct \{productName}")
      .withColumn("Amount", FloatType(), 
                  minValue=pVars["min"],maxValue=pVars["max"], 
                  distribution=dist.Normal(mean=pVars["mean"], stddev=.001), random=True)
      .withColumn("Label", IntegerType(), minValue=0, maxValue=0)).build()
    

# Generate a record
def generateRecord(context,Product_vars, Label, CustomerID_vars):
  product = get_random_product(Product_vars)
  return (define_specs(context,productName=product, pVars=Product_vars[product], Label=Label, CustomerID_vars=CustomerID_vars, currentTimestamp=datetime.now()))

import random

def get_random_product(Product_vars):
  products = list(Product_vars.keys())
  return products[random.randint(0,len(products)-1)]

from pyspark.sql.functions import expr
from functools import reduce
import pyspark
def returnTransactionDf(context,Product_vars, CustomerID_vars):
  recordSet = []
  numRecords = random.randint(1,10)
  for record in range(numRecords):
    recordSet.append(generateRecord(context,Product_vars=Product_vars,Label=random.randint(0,1), CustomerID_vars=CustomerID_vars))
  recordDF = reduce(pyspark.sql.dataframe.DataFrame.unionByName, recordSet) 
  recordDF = recordDF.withColumn("Amount", expr("Amount / 100"))
  return recordDF

from pyspark.dbutils import DBUtils
import os
# Generate a set of data, convert it to a Dataframe, write it out as one json file to the temp path. Then move that file to the destination_path
def writeJsonFile(context,destination_path,temp_path, Product_vars, CustomerID_vars):
  dbutils = DBUtils(context)
  recordDF = returnTransactionDf(context,Product_vars, CustomerID_vars)
  recordDF.coalesce(1).write.mode("overwrite").format("json").save(temp_path)
  
  # Grab the file from the temp location, write it to the location we want and then delete the temp directory
  tempJson = os.path.join(temp_path, dbutils.fs.ls(temp_path)[3][1])
  dbutils.fs.cp(tempJson, destination_path)
  dbutils.fs.rm(temp_path, True)

