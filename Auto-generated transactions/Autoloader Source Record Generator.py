# Databricks notebook source
# MAGIC %md
# MAGIC ## Generate a JSON dataset for the Autoloader to pick up

# COMMAND ----------

# DBTITLE 1,Define Record Count, Temporary Location, Autoloader-Monitored Location and Sleep Interval Here
recordCount=5
nIDs = 10
dbutils.widgets.text("stemFilePath",  "s3://oetrta/ss-2-fs/stephanie.rivera")
tempPath = "{}/temp".format(dbutils.widgets.get("stemFilePath"))
destinationPath = "{}/autoloaderinput/".format(dbutils.widgets.get("stemFilePath"))
sleepIntervalSeconds = 3 

# COMMAND ----------

# DBTITLE 1,Reset Environment & Setup
dbutils.fs.rm(tempPath, recurse=True)
dbutils.fs.rm(destinationPath, recurse=True)
dbutils.fs.mkdirs(destinationPath)

# COMMAND ----------

# DBTITLE 1,Functions to generate a JSON dataset for the Autoloader to pick up
import random
import string
from datetime import datetime
import time
import os

# Method to return a random User ID between 1 and 5 (set low for testing some stateful streaming aggregations, higher for more variability)
def returnCustomerID(nIDs):
  return random.randint(1, nIDs)

# Return a random float value for different purposes, rounded to 2 places
def returnValue():
  return round(random.uniform(2.11, 399.99), 2)

# Method to return a Product string
def returnString():
  letters = string.ascii_letters
  return ('Product ' + ''.join(random.choice(letters.upper()) for i in range(1)) )

def returnTransactionTimestamp():
  currentDateTime = datetime.now()
  return currentDateTime.strftime("%Y-%m-%d %H:%M:%S.%f")

# Generate a record
def generateRecord(nIDs):
  return (returnCustomerID(nIDs), returnString(), returnValue(), returnTransactionTimestamp())
  
# Generate a list of records
def generateRecordSet(recordCount, nIDs):
  recordSet = []
  for x in range(recordCount):
    recordSet.append(generateRecord(nIDs))
  return recordSet

# Generate a set of data, convert it to a Dataframe, write it out as one json file in a temp location, 
# move the json file to the desired location that the autoloader will be watching and then delete the temp location
def writeJsonFile(recordCount, nIDs, tempPath, destinationPath):
  recordColumns = ["CustomerID", "Product", "Amount", "TransactionTimestamp"]
  recordSet = generateRecordSet(recordCount,nIDs)
  recordDf = spark.createDataFrame(data=recordSet, schema=recordColumns)
  
  # Write out the json file with Spark in a temp location - this will create a directory with the file we want
  recordDf.coalesce(1).write.format("json").save(tempPath)
  
  # Grab the file from the temp location, write it to the location we want and then delete the temp directory
  tempJson = os.path.join(tempPath, dbutils.fs.ls(tempPath)[3][1])
  dbutils.fs.cp(tempJson, destinationPath)
  dbutils.fs.rm(tempPath, True)

# COMMAND ----------

# DBTITLE 1,Loop for Generating Data
t=1
while(t<20):
  writeJsonFile(recordCount, nIDs, tempPath, destinationPath)
  t = t+1
  time.sleep(sleepIntervalSeconds)

# COMMAND ----------

# DBTITLE 1,Count of Transactions per User
df = spark.read.format("json").load(destinationPath)
usercounts = df.groupBy("CustomerID").count()
display(usercounts.orderBy("CustomerID"))


# COMMAND ----------

# DBTITLE 1,Display the Data Generated
display(df)
