// Databricks notebook source
// DBTITLE 1,Aggregating transactions for each customerID
case class InputRow(CustomerID: Long, 
                    TransactionTimestamp: java.time.Instant)

case class TransactionCountState(latestTimestamp: java.time.Instant,
                                  currentTransactions: List[InputRow])

case class TransactionCount(CustomerID: Long, 
                            transactionCount: Integer, 
                            eventTimestamp: java.time.Instant, 
                            isTimeout: Boolean) 


def addNewRecords(newRecords: List[InputRow], 
                  transactionCountState: TransactionCountState): TransactionCountState = 
  {
    val recordWithLatestTimestamp = 
        newRecords.maxBy(record => record.TransactionTimestamp)
    val latestNewTimestamp = recordWithLatestTimestamp.TransactionTimestamp
    val latestTimestamp = 
        if (latestNewTimestamp.toEpochMilli() > transactionCountState.latestTimestamp.toEpochMilli()
        ) latestNewTimestamp else transactionCountState.latestTimestamp

    new TransactionCountState(latestTimestamp, 
        transactionCountState.currentTransactions ::: newRecords)
  }


def dropExpiredRecords(newLatestTimestamp: java.time.Instant, 
    currentTransactions: List[InputRow]): TransactionCountState = {
    val newTransactionList = ListBuffer[InputRow]()
    val expirationTimestamp = Instant.ofEpochMilli(newLatestTimestamp.toEpochMilli() - TimeUnit.MINUTES.toMillis(windowMinutes))

    if (currentTransactions.size > 0) {
      currentTransactions.foreach { value => 
        if (value.TransactionTimestamp.toEpochMilli() >= expirationTimestamp.toEpochMilli())
          newTransactionList.append(value)
      }
    }
    
    new TransactionCountState(newLatestTimestamp, 
                              newTransactionList.toList)
  }

def updateState(
  CustomerID: Long,  
  values: Iterator[InputRow],  
  state: GroupState[TransactionCountState]): Iterator[TransactionCount] = {  
 
  val transactionCounts = ListBuffer[TransactionCount]()
  
  if (!state.hasTimedOut) {
    val transactionList = new ListBuffer[InputRow]()
    values.foreach { value =>
      transactionList.append(value)}

    var prevState = state.getOption.getOrElse {
      val firstTransactionTimestamp = transactionList.head.TransactionTimestamp
      new TransactionCountState(firstTransactionTimestamp, List[InputRow]())}

    val stateWithNewRecords = addNewRecords(transactionList.toList, 
      prevState)
    
    val stateWithRecordsDropped = 
      dropExpiredRecords(stateWithNewRecords.latestTimestamp, 
      stateWithNewRecords.currentTransactions)
      
    val output = new TransactionCount(CustomerID, 
      stateWithRecordsDropped.currentTransactions.size, 
      stateWithRecordsDropped.latestTimestamp, false)

    transactionCounts.append(output)
    state.update(stateWithRecordsDropped)
    state.setTimeoutTimestamp(
      stateWithRecordsDropped.latestTimestamp.toEpochMilli(), 
      "30 seconds")
  } else {
    val prevState = state.get
    val newTimestamp = Instant.now
    val stateWithRecordsDropped = dropExpiredRecords(newTimestamp,   
      prevState.currentTransactions)
    val output = new TransactionCount(CustomerID, 
      stateWithRecordsDropped.currentTransactions.size, 
      stateWithRecordsDropped.latestTimestamp, true)
    
    transactionCounts.append(output)
    state.update(stateWithRecordsDropped)
    state.setTimeoutTimestamp(
      stateWithRecordsDropped.latestTimestamp.toEpochMilli(), 
      "30 seconds")
  }

  transactionCounts.toIterator
}

// COMMAND ----------

// DBTITLE 1,Schema for checking type
import org.apache.spark.sql.types.{StringType, 
            StructField, StructType, LongType, 
            DoubleType, TimestampType}

val schema = StructType(Array(
              StructField("Source", StringType, true),
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

// Read from table we created using Auto-Loader
val inputDf =
  spark.readStream
    .format("delta")
    .schema(schema)
    .table(inputTable)
    .selectExpr("CustomerID", "cast(TransactionTimestamp as timestamp) TransactionTimestamp")
    .as[InputRow]

// Execute flatMapGroupsWithState
// We're allowing data to be 30 seconds late before it is dropped
val flatMapGroupsWithStateResultDf = 
  inputDf
    .withWatermark("TransactionTimestamp", "30 seconds")
    .groupByKey(_.CustomerID)
    .flatMapGroupsWithState(OutputMode.Append, GroupStateTimeout.EventTimeTimeout)(updateState)

// Function for foreachBatch to update the counts in the Delta table
def updateCounts(newCountsDs: Dataset[TransactionCount], epoch_id: Long): Unit = {
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

// Save the flatMapGroupsWithState result to a Delta table.  Delta tables do not support streaming updates directly, so we need to use a foreachBatch function
flatMapGroupsWithStateResultDf.writeStream
  .foreachBatch(updateCounts _)
  .option("checkpointLocation", f"$outputPath/checkpoint")
  .trigger(Trigger.ProcessingTime("10 seconds"))
  .queryName("flatMapGroups")  //query name will show up in Spark UI
  .start()

// COMMAND ----------

// DBTITLE 1,Writing the history of transaction counts to a table
flatMapGroupsWithStateResultDf.writeStream
  .option("checkpointLocation", f"$outputPath/checkpoint2")
  .trigger(Trigger.ProcessingTime("10 seconds"))
  .queryName("flatMapGroupsHistoryTable")  //query name will show up in Spark UI
  .table(history_table_name)

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from transaction_count_ft order by eventTimestamp desc

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT * FROM transaction_count_history LIMIT 10

// COMMAND ----------


