// Databricks notebook source
// MAGIC %md
// MAGIC Chapter 5: Feature Engineering
// MAGIC
// MAGIC ##Synthetic data - Feature engineering on a streaming table
// MAGIC

// COMMAND ----------

// MAGIC %python
// MAGIC dbutils.widgets.dropdown(name='Reset', defaultValue='False', choices=['True', 'False'], label="Reset Checkpoint and Schema")

// COMMAND ----------

// MAGIC %md ## Run Setup

// COMMAND ----------

// MAGIC %run ../../global-setup $project_name=synthetic_transactions 

// COMMAND ----------

// DBTITLE 1,Configurations
import java.time.Instant
import java.util.concurrent.TimeUnit
import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.streaming.{GroupStateTimeout, OutputMode, GroupState}

// These can also be in the cluster settings - it will automatically compact sets of small files into larger files as it writes for more optimal read performance
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

// COMMAND ----------

// DBTITLE 1,Resets
// MAGIC %python
// MAGIC table_name = "transaction_count_ft"
// MAGIC history_table_name = "transaction_count_history"
// MAGIC if bool(dbutils.widgets.get('Reset')):
// MAGIC   dbutils.fs.rm(f"{volume_file_path}/{table_name}/table_feature_outputs/", True)
// MAGIC   dbutils.fs.rm(f"{volume_file_path}/{history_table_name}/table_feature_outputs/", True)
// MAGIC   sql(f"DROP TABLE IF EXISTS {table_name}")
// MAGIC   sql(f"DROP TABLE IF EXISTS {history_table_name}")

// COMMAND ----------

// DBTITLE 1,Set up table, paths, and variables
// variables passed from the setup file are in python
val table_name = "transaction_count_ft"
val history_table_name = "transaction_count_history"
val volumePath = "/Volumes/ml_in_action/synthetic_transactions/files/"
val outputPath = f"$volumePath/$table_name/streaming_outputs/"
val outputPath2 = f"$volumePath/$history_table_name/streaming_outputs/"
val inputTable = "ml_in_action.synthetic_transactions.raw_transactions"

//// maybe we need a try catch here
sql(f"""CREATE OR REPLACE TABLE $table_name (CustomerID Int, transactionCount Int, eventTimestamp Timestamp, isTimeout Boolean)""")
sql(f"""ALTER TABLE $table_name ALTER COLUMN CustomerID SET NOT NULL""")
sql(f"""ALTER TABLE $table_name ADD PRIMARY KEY(CustomerID)""")
sql(f"ALTER TABLE $table_name SET TBLPROPERTIES (delta.enableChangeDataFeed=true)")


sql(f"""CREATE OR REPLACE TABLE $history_table_name (CustomerID Int, transactionCount Int, eventTimestamp Timestamp, isTimeout Boolean)""")
sql(f"""ALTER TABLE $history_table_name ALTER COLUMN CustomerID SET NOT NULL""")
sql(f"""ALTER TABLE $history_table_name ALTER COLUMN eventTimestamp SET NOT NULL""")
sql(f"""ALTER TABLE $history_table_name ADD PRIMARY KEY(CustomerID, eventTimestamp TIMESERIES)""")

// aggregate transactions for windowMinutes
val windowMinutes = 2
// wait at most max_wait_minutes before writing a record
val maxWaitMinutes = 1

// COMMAND ----------

// MAGIC %python
// MAGIC from databricks.feature_engineering import FeatureEngineeringClient
// MAGIC fe = FeatureEngineeringClient()
// MAGIC
// MAGIC fe.set_feature_table_tag(name=f"{table_name}", key="FE_role", value="online_serving")
// MAGIC fe.set_feature_table_tag(name=f"{history_table_name}", key="FE_role", value="training_data")

// COMMAND ----------

// DBTITLE 1,Aggregating transactions for each customerID
// The structure of the input data - a user and a transaction
case class InputRow(CustomerID: Integer, 
                    TransactionTimestamp: java.time.Instant)

// This is what the stream is storing so that it can count the number of transactions for a customer within the last window minutes
case class TransactionCountState(latestTimestamp: java.time.Instant, 
                                  currentTransactions: List[InputRow])

// The structure of the values being emitted - includes the event datetime that triggered this update and a boolean indicating whether the update was triggered by a timeout, meaning a record wasn't received for a customer in a minute
case class TransactionCount(CustomerID: Integer, 
                            transactionCount: Integer, 
                            eventTimestamp: java.time.Instant, 
                            isTimeout: Boolean) 

def addNewRecords(newRecords: List[InputRow], transactionCountState: TransactionCountState): TransactionCountState = {
  // Get the latest timestamp in the set of new records
  val recordWithLatestTimestamp = newRecords.maxBy(record => record.TransactionTimestamp)
  val latestNewTimestamp = recordWithLatestTimestamp.TransactionTimestamp
  
  // Compare to the latestTimestamp in the transactionCountState, use whichever is greater
  // This is in case we've received data out of order
  val latestTimestamp = if (latestNewTimestamp.toEpochMilli() > transactionCountState.latestTimestamp.toEpochMilli()) latestNewTimestamp else transactionCountState.latestTimestamp
  
  // Create a new TransactionCountState object with the latest timestamp and combining the two record lists and return
  new TransactionCountState(latestTimestamp, transactionCountState.currentTransactions ::: newRecords)
}

// Drop records that are more than windowMinutes old
def dropExpiredRecords(newLatestTimestamp: java.time.Instant, currentTransactions: List[InputRow]): TransactionCountState = {
  val newTransactionList = ListBuffer[InputRow]()
  
  // Calculate the state expiration timestamp
  val expirationTimestamp = Instant.ofEpochMilli(newLatestTimestamp.toEpochMilli() - TimeUnit.MINUTES.toMillis(windowMinutes))
  
  // If there are records in state, loop through the list of current transactions and keep any that are before their expiration timestamp
  if (currentTransactions.size > 0) {
    currentTransactions.foreach { value => 
      if (value.TransactionTimestamp.toEpochMilli() >= expirationTimestamp.toEpochMilli())
        newTransactionList.append(value)
    }
  }
  // Create new TransactionCountState object and return
  new TransactionCountState(newLatestTimestamp, newTransactionList.toList)
}

def updateState(
  CustomerID: Integer,
  values: Iterator[InputRow],
  state: GroupState[TransactionCountState]): Iterator[TransactionCount] = {
  
  // Create a new ListBuffer to store what we're going to return at the end of processing this key
  val transactionCounts = ListBuffer[TransactionCount]()
  
  if (!state.hasTimedOut) {
    val transactionList = new ListBuffer[InputRow]()
    values.foreach { value =>
      transactionList.append(value)
    }

    var prevState = state.getOption.getOrElse {
      val firstTransactionTimestamp = transactionList.head.TransactionTimestamp
      new TransactionCountState(firstTransactionTimestamp, List[InputRow]())
    }
    
    val stateWithNewRecords = addNewRecords(transactionList.toList, prevState)
    
    val stateWithRecordsDropped = dropExpiredRecords(stateWithNewRecords.latestTimestamp, stateWithNewRecords.currentTransactions)
    
    val output = new TransactionCount(CustomerID, stateWithRecordsDropped.currentTransactions.size, stateWithRecordsDropped.latestTimestamp, false)
    transactionCounts.append(output)
    
    // Save the state
    state.update(stateWithRecordsDropped)
    state.setTimeoutTimestamp(stateWithRecordsDropped.latestTimestamp.toEpochMilli(), "30 seconds")
  } else {

    val prevState = state.get
    val newTimestamp = Instant.now
    
    val stateWithRecordsDropped = dropExpiredRecords(newTimestamp, prevState.currentTransactions)

    val output = new TransactionCount(CustomerID, stateWithRecordsDropped.currentTransactions.size, stateWithRecordsDropped.latestTimestamp, true)
    transactionCounts.append(output)
    
    // Save the state
    state.update(stateWithRecordsDropped)
    
    // Set the timeout to now plus 30 seconds.  If no data is seen for this key in the next 30 seconds then this function will be triggered again to drop expired records and emit a count
    // Since the watermark is set at 30 seconds then this timeout will trigger approximately once per minute
    state.setTimeoutTimestamp(stateWithRecordsDropped.latestTimestamp.toEpochMilli(), "30 seconds")
  }
  
  // Return an iterator of records from flatMapGroupsWithState.  
  transactionCounts.toIterator
}

// COMMAND ----------

// DBTITLE 1,Schema for checking type
import org.apache.spark.sql.types.{StringType, 
          StructField, StructType, IntegerType, 
          FloatType, TimestampType}

// The schema for the incoming records
val schema = StructType(Array(
              StructField("Source", StringType, true),
              StructField("TransactionTimestamp", StringType, true),
              StructField("CustomerID", IntegerType, true),
              StructField("Amount", FloatType, true),
              StructField("Product", StringType, true),
              StructField("Label", IntegerType, true),
                              ))

// COMMAND ----------

// DBTITLE 1,Read and write streams
import io.delta.tables._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.streaming.Trigger


val inputDf =
  spark.readStream
    .format("delta")
    .schema(schema)
    .table(inputTable)
    .selectExpr("CustomerID", "cast(TransactionTimestamp as timestamp) TransactionTimestamp")
    .as[InputRow]

// We're allowing data to be 30 seconds late before it is dropped
val flatMapGroupsWithStateResultDf = 
  inputDf
    .withWatermark("TransactionTimestamp", "30 seconds")
    .groupByKey(_.CustomerID)
    .flatMapGroupsWithState(OutputMode.Append, 
        GroupStateTimeout.EventTimeTimeout)(updateState)

def updateCounts(newCountsDs: Dataset[TransactionCount], 
                  epoch_id: Long): Unit = {
  // Convert the dataset to a dataframe for merging
  val newCountsDf = newCountsDs.toDF
  
  // Get the target Delta table that is being merged
  val aggregationTable = DeltaTable.forName(table_name)

  // Merge the new records into the target Delta table.
  aggregationTable.alias("t")
    .merge(newCountsDf.alias("m"), "m.CustomerID = t.CustomerID")
    .whenMatched().updateAll()
    .whenNotMatched().insertAll()
    .execute()
}

// Save the flatMapGroupsWithState result to a Delta table.  
flatMapGroupsWithStateResultDf.writeStream
  .foreachBatch(updateCounts _)
  .option("checkpointLocation", f"$outputPath/checkpoint")
  .trigger(Trigger.ProcessingTime("10 seconds"))
  .queryName("flatMapGroups")
  .start()

// COMMAND ----------

// DBTITLE 1,Writing the history of transaction counts to a table
flatMapGroupsWithStateResultDf.writeStream
  .option("checkpointLocation", f"$outputPath2/checkpoint")
  .trigger(Trigger.ProcessingTime("10 seconds"))
  .queryName("flatMapGroupsHistoryTable")
  .table(history_table_name)

// COMMAND ----------


