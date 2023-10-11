// Databricks notebook source
// MAGIC %md
// MAGIC #Synthetic data
// MAGIC
// MAGIC ##Run setup

// COMMAND ----------

// MAGIC %python
// MAGIC dbutils.widgets.dropdown(name='Reset', defaultValue='False', choices=['True', 'False'], label="Reset Checkpoint and Schema")

// COMMAND ----------

// MAGIC %run ../../global-setup $project_name=synthetic_data $catalog=hive_metastore

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
// MAGIC if bool(dbutils.widgets.get('Reset')):
// MAGIC   dbutils.fs.rm(f"{cloud_storage_path}/table_feature_outputs/", True)
// MAGIC   sql("DROP TABLE IF EXISTS synthetic_streaming_features")

// COMMAND ----------

// DBTITLE 1,Set up table, paths, and variables
// variables passed from the setup file are in python
val sparkStoragePath = "s3://one-env/lakehouse_ml_in_action/synthetic_data/"
val outputPath = f"$sparkStoragePath/table_feature_outputs/"
val inputTable = "synthetic_transactions"

val table_name = "synthetic_streaming_features"
val catalog = "hive_metastore"
val database = "lakehouse_in_action"
sql(f"""CREATE OR REPLACE TABLE $table_name (CustomerID Long, transactionCount Int, eventTimestamp Timestamp, isTimeout Boolean)""")
sql(f"ALTER TABLE $table_name SET TBLPROPERTIES (delta.enableChangeDataFeed=true)")

// aggregate transactions for windowMinutes
val windowMinutes = 2
// wait at most max_wait_minutes before writing a record
val maxWaitMinutes = 1

// COMMAND ----------

// DBTITLE 1,Aggregating transactions for each customerID
// The case class for the structure of the input data - a user and a transaction
case class InputRow(CustomerID: Long, TransactionTimestamp: java.time.Instant)

// The case class for the state - this is what the stream is storing so that it can count the number of transactions for a user within the last 5 minutes
case class TransactionCountState(latestTimestamp: java.time.Instant, currentTransactions: List[InputRow])

// The case class for the values being emitted - the key (user), the count of transactions in the last 5 minutes, the event datetime that triggered this update and a boolean 
// indicating whether the update was triggered by a timeout, meaning a record wasn't received for the user within a minute
case class TransactionCount(CustomerID: Long, transactionCount: Integer, eventTimestamp: java.time.Instant, isTimeout: Boolean) 

// Add new records to the state
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
  
  // Calculate the state expiration timestamp - the latest timestamp minus the transaction count minutes
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

// This is the function that is called with flatMapGroupsWithState.  It keeps track of the last 5 minutes of records for each key so that each time new data is received
// it can count the number of transactions that occurred in the last 5 minuts.
// This function will be called in two ways -
//   If one or more records for a given user are received.  In that case it will add those records to the state, drop any records that are older than 5 minuts from the state and calculate the count
//   If no records are received for a given user within a minute since the last time this function was called.  In that case it will drop any records that are older than 5 minutes from the state and calculate the count
def updateState(
  CustomerID: Long,  // This is the key we are grouping on
  values: Iterator[InputRow],  // This is the format of the records coming into the function
  state: GroupState[TransactionCountState]): Iterator[TransactionCount] = {   // This declares the format of the state records we're storing and the format of the records we're outputting
  
  // Create a new ListBuffer to store what we're going to return at the end of processing this key
  val transactionCounts = ListBuffer[TransactionCount]()
  
  // If we haven't timed out then there are values for this key
  if (!state.hasTimedOut) {
    // There can be one or more records for this key.  Iterate through them and put them in a list 
    val transactionList = new ListBuffer[InputRow]()
    values.foreach { value =>
      transactionList.append(value)
    }

    // Now get the previous state if it exists.  If it doesn't exist (if this is the first time we've received a record for this user the state won't exist yet) then set the initial state to the 
    // TransactionTimestamp of the first input record and an empty List of InputRow
    var prevState = state.getOption.getOrElse {
      val firstTransactionTimestamp = transactionList.head.TransactionTimestamp
      new TransactionCountState(firstTransactionTimestamp, List[InputRow]())
    }
    
    // Add the new records to the state
    val stateWithNewRecords = addNewRecords(transactionList.toList, prevState)
    
    // Drop expired records from the state
    // After this function only the transactions that occurred within the last five minutes from the latest transsaction will be in the state
    val stateWithRecordsDropped = dropExpiredRecords(stateWithNewRecords.latestTimestamp, stateWithNewRecords.currentTransactions)
    
    // Create the output record - the key (user), count of transactions in the last 5 minutes, the latest timestamp and a boolean indicating this record was not triggered by a timeout
    val output = new TransactionCount(CustomerID, stateWithRecordsDropped.currentTransactions.size, stateWithRecordsDropped.latestTimestamp, false)
    transactionCounts.append(output)
    
    // Save the state
    state.update(stateWithRecordsDropped)
    
    // When no data has been seen for a period of time for a given key, this timeout will trigger the else clause below
    // The timeout will only trigger after the watermark has moved pasted this timestamp.  So for example if we're allowing data to be up to 30 seconds late,
    // then this timeout will trigger at the configured timestamp plus 30 seconds
    // Set the timeout to the latest TransactionTimestamp that's in state plus 30 seconds.  If no data is seen
    // for 30 seconds past the latest TransactionTimestamp in the state then this function will be triggered again to drop expired records and emit a count
    // Since the watermark is set at 30 seconds then this timeout will trigger approximately once per minute
    state.setTimeoutTimestamp(stateWithRecordsDropped.latestTimestamp.toEpochMilli(), "30 seconds")
  } else {
    // Since a timeout was triggered that means there was no input for this key
    // Use now as the new timestamp for the state
    // Drop expired records from state
    // Create the new output record (if all the transactions were dropped then the count will be 0)
    // Set the new state
    // Set the timeout
    val prevState = state.get
    val newTimestamp = Instant.now
    
    // Drop expired records from the state
    // After this function only the transactions that occurred within the last five minutes from the latest transsaction will be in state
    val stateWithRecordsDropped = dropExpiredRecords(newTimestamp, prevState.currentTransactions)
    
    // Create the output record - the key (user), count of transactions in the last 5 minutes, the latest timestamp and a boolean indicating this record was triggered by a timeout
    val output = new TransactionCount(CustomerID, stateWithRecordsDropped.currentTransactions.size, stateWithRecordsDropped.latestTimestamp, true)
    transactionCounts.append(output)
    
    // Save the state
    state.update(stateWithRecordsDropped)
    
    // Set the timeout to now plus 30 seconds.  If no data is seen for this key in the next 30 seconds then this function will be triggered again to drop expired records and emit a count
    // Since the watermark is set at 30 seconds then this timeout will trigger approximately once per minute
    state.setTimeoutTimestamp(stateWithRecordsDropped.latestTimestamp.toEpochMilli(), "30 seconds")
  }
  
  // Return an iterator of records from flatMapGroupsWithState.  In this use case the output will contain one record
  // It is valid to return an empty Iterator from flatMapGroupsWithState
  transactionCounts.toIterator
}

// COMMAND ----------

// DBTITLE 1,Schema for checking type
import org.apache.spark.sql.types.{StringType, StructField, StructType, LongType, DoubleType, TimestampType}

// The schema for the incoming records
val schema = StructType(Array(StructField("Source", StringType, true),
                 StructField("TransactionTimestamp", StringType, true),
                 StructField("CustomerID", LongType, true),
                 StructField("Amount", DoubleType, true),
                 StructField("Product", StringType, true)
                                 ))

// COMMAND ----------

// DBTITLE 1,Read and write streams
import io.delta.tables._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.streaming.Trigger

// Read from cloud storage using the Auto-Loader
// There are two ways to use the Auto-Loader - Directory listing and Notifications.  Directory listing is the default, and only requires permissions on the cloud bucket that you want to read
// When a new stream is first set up using the useNotifications option, the Auto-Loader automatically spins up cloud resources that get events from the input directory
val inputDf =
  spark.readStream
    .format("delta")
    .schema(schema)
    .table(inputTable)
    .selectExpr("CustomerID", "cast(TransactionTimestamp as timestamp) TransactionTimestamp")
    .as[InputRow]  // Specifically set the type of the Dataframe to the case class that flatMapGroupsWithState is expecting

// Execute flatMapGroupsWithState and write out to Delta
// We're allowing data to be 30 seconds late before it is dropped
val flatMapGroupsWithStateResultDf = 
  inputDf
    .withWatermark("TransactionTimestamp", "30 seconds")
    .groupByKey(_.CustomerID)
    .flatMapGroupsWithState(OutputMode.Append, GroupStateTimeout.EventTimeTimeout)(updateState)

// Function for foreachBatch to update the counts in the Delta table
def updateCounts(newCountsDs: Dataset[TransactionCount], epoch_id: Long): Unit = {
  // Convert the dataset (which was output by the mapGroupsWithState function) to a dataframe for merging
  val newCountsDf = newCountsDs.toDF
  
  // Get the target Delta table that is being merged
  val aggregationTable = DeltaTable.forName(table_name)

  // Merge the new records into the target Delta table.  This can be done with SQL syntax as well
  aggregationTable.alias("t")
    .merge(newCountsDf.alias("m"), "m.CustomerID = t.CustomerID")
    .whenMatched().updateAll()
    .whenNotMatched().insertAll()
    .execute()
}

// Save the flatMapGroupsWithState result to a Delta table.  Delta tables do not support streaming updates directly, so we need to use a foreachBatch function
flatMapGroupsWithStateResultDf.writeStream
  .foreachBatch(updateCounts _)
  .option("checkpointLocation", f"$outputPath/checkpoint")
  .trigger(Trigger.ProcessingTime("10 seconds"))
  .queryName("flatMapGroups")  //query name will show up in Spark UI
  .start()

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from lakehouse_in_action.synthetic_streaming_features order by eventTimestamp desc;

// COMMAND ----------


