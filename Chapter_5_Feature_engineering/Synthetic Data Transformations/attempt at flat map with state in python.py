# Databricks notebook source
# MAGIC %run ../global-setup $project_name=transactional_data

# COMMAND ----------

# DBTITLE 1,Setting variables
import os
import shutil
import math
import time
from datetime import datetime
from typing import Tuple, Iterator
from pathlib import Path
from functools import reduce

import pandas as pd

from pyspark.sql.functions import current_timestamp
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

table_name = "transactional_data_features"

output_path = f"{cloud_storage_path}/feature_outputs/"

# aggregate transactions for window_minutes
window_minutes = 2
# wait at most max_wait_minutes before writing a record
max_wait_minutes = 1

# input transaction - a customer id,

# COMMAND ----------

# DBTITLE 1,Spark optimizations
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", True)
spark.conf.set("spark.databricks.delta.autoCompact.enabled", True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Our goals
# MAGIC 1. This logic will keep track of all the transactions that occurred in the previous 5 minutes for a given user and update the count every time new transactions are received for that user
# MAGIC 2. If no transactions have been received after 1 minute for a given user, the logic will still emit a count and will remove records from state that are more than 5 minutes old
# MAGIC 3. If the stream has no data coming through at all then nothing will be updated. Something must be coming through the stream for this logic to be executed
# MAGIC 4. The transactionCountMinutes and maxRecordIntervalMinutes variables can be updated below to change how far back in time to count records and how often a new count will be emitted if no new records for a user are received

# COMMAND ----------

transactions = spark.readStream \
  .format("cloudFiles") \
  .option("cloudFiles.format", "json") \
  .option("cloudFiles.schemaHints","CustomerID bigint, Amount double, TransactionTimestamp timestamp") \
  .option("cloudFiles.schemaEvolutionMode", "none") \
  .option("cloudFiles.schemaLocation", schema_location) \
  .load(raw_data_location) \
  .select("*") \
  



# COMMAND ----------

def timeInstant():
  return int(datetime.utcnow().timestamp() * 1000)

def removeExpiredTransactions()

def featureWithState(
    customer_id: bigint, pdfs: Iterator[pd.DataFrame], state: GroupState
) -> Iterator[pd.DataFrame]:
    if state.hasTimedOut:
        (word,) = customer_id
        (count,) = state.get
        state.remove()
        yield pd.DataFrame({"session": [word], "count": [count]})
    else:
        # Aggregate the number of words.
        count = sum(map(lambda pdf: len(pdf), pdfs))
        if state.exists:
            (old_count,) = state.get
            count += old_count
        state.update((count,))
        # Set the timeout as 10 seconds.
        state.setTimeoutDuration(10000)
        yield pd.DataFrame()

# COMMAND ----------

.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", checkpoint_location) \
  .option("mergeSchema", "true") \
  .trigger(processingTime='10 seconds') \
  .toTable(tableName=table_name)

# COMMAND ----------

int(datetime.utcnow().timestamp() * 1000)

# COMMAND ----------


