import dbldatagen as dg
from datetime import datetime
import dbldatagen.distributions as dist
from pyspark.sql.types import IntegerType, FloatType, StringType

def define_specs(context, Product, Label, currentTimestamp):
  CustomerID_vars = {"min": 1234, "max": 1260}

  Product_vars = {1: {
                  "A": {"min": 1000, "max": 25037, "mean": 15520, "alpha": 10, "beta": 25},
                  "B": {"min": 1000, "max": 5141, "mean": 35520, "alpha": 8, "beta": 2},
                  "C": {"min": 10000, "max": 41473, "mean": 30520, "alpha": 2, "beta": 4}},
                  0: {
                  "A": {"min": 1000, "max": 25001, "mean": 15520, "alpha": 1, "beta": 100},
                  "B": {"min": 1000, "max": 5501, "mean": 35520, "alpha": 10, "beta": 4},
                  "C": {"min": 10000, "max": 40001, "mean": 30520, "alpha": 3, "beta": 10}}}

  pVars = Product_vars[Label][Product]
  return (dg.DataGenerator(context, name="syn_trans", rows=1, partitions=4, randomSeed=-1)
    .withColumn("CustomerID", IntegerType(), nullable=False,
                minValue=CustomerID_vars["min"], maxValue=CustomerID_vars["max"], random=True)
    .withColumn("TransactionTimestamp", "timestamp", random=False,
                begin=currentTimestamp, end=currentTimestamp,nullable=False)
    .withColumn("Product", StringType(), template=f"Pro\duct \{Product}")
    .withColumn("Amount", FloatType(), 
                minValue=pVars["min"],maxValue=pVars["max"], 
                distribution=dist.Beta(alpha=pVars["alpha"], beta=pVars["beta"]), random=True)
    .withColumn("Label", IntegerType(), minValue=Label, maxValue=Label)).build()
  
import random
from pyspark.sql.functions import expr
from functools import reduce
import pyspark

def returnTransactionDf(context):
  products = ["A","B","C"]
  recordSet = []
  numRecords = random.randint(1,10)
  for record in range(numRecords):
    recordSet.append(define_specs(context, Product=products[random.randint(0,len(products)-1)],Label=random.randint(0,1),currentTimestamp = datetime.now()))
  recordDF = reduce(pyspark.sql.dataframe.DataFrame.unionByName, recordSet) 
  recordDF = recordDF.withColumn("Amount", expr("Amount / 100"))
  return recordDF

from pyspark.dbutils import DBUtils
import os
# Generate a set of data, convert it to a Dataframe, write it out as one json file to the temp path. Then move that file to the destination_path
def writeJsonFile(context,record_path,label_path,temp_path):
  dbutils = DBUtils(context)
  recordDF = returnTransactionDf(context=context)
  recordDF.select("Amount","CustomerID","Product","TransactionTimestamp").coalesce(1).write.mode("overwrite").format("json").save(temp_path)
  tempJson = os.path.join(temp_path, dbutils.fs.ls(temp_path)[3][1])
  dbutils.fs.cp(tempJson, record_path)
  dbutils.fs.rm(temp_path, True)

  # Grab the file from the temp location, write it to the location we want and then delete the temp directory
  recordDF.select("CustomerID","TransactionTimestamp","Label").coalesce(1).write.mode("overwrite").format("json").save(temp_path)
  tempJson = os.path.join(temp_path, dbutils.fs.ls(temp_path)[3][1])
  dbutils.fs.cp(tempJson, label_path)
  dbutils.fs.rm(temp_path, True)
