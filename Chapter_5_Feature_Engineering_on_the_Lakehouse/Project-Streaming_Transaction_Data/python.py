# Databricks notebook source
import io.delta.tables as dt
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, LongType, DoubleType, TimestampType

class InputRow:
    def __init__(self, CustomerID, TransactionTimestamp):
        self.CustomerID = CustomerID
        self.TransactionTimestamp = TransactionTimestamp

class TransactionCountState:
    def __init__(self, latestTimestamp, currentTransactions):
        self.latestTimestamp = latestTimestamp
        self.currentTransactions = currentTransactions

class TransactionCount:
    def __init__(self, CustomerID, transactionCount, eventTimestamp, isTimeout):
        self.CustomerID = CustomerID
        self.transactionCount = transactionCount
        self.eventTimestamp = eventTimestamp
        self.isTimeout = isTimeout

def add_new_records(new_records, transaction_count_state):
    record_with_latest_timestamp = max(new_records, key=lambda record: record.TransactionTimestamp)
    latest_new_timestamp = record_with_latest_timestamp.TransactionTimestamp

    latest_timestamp = latest_new_timestamp if latest_new_timestamp > transaction_count_state.latestTimestamp else transaction_count_state.latestTimestamp

    return TransactionCountState(latest_timestamp, transaction_count_state.currentTransactions + new_records)

def drop_expired_records(new_latest_timestamp, current_transactions):
    new_transaction_list = []
    expiration_timestamp = new_latest_timestamp - F.interval(2, "minutes")

    for transaction in current_transactions:
        if transaction.TransactionTimestamp >= expiration_timestamp:
            new_transaction_list.append(transaction)

    return TransactionCountState(new_latest_timestamp, new_transaction_list)

def update_state(customer_id, values, state):
    transaction_counts = []

    if not state.hasTimedOut:
        transaction_list = []
        for value in values:
            transaction_list.append(value)

        prev_state = state.getOption() or TransactionCountState(transaction_list[0].TransactionTimestamp, [])

        state_with_new_records = add_new_records(transaction_list, prev_state)
        state_with_records_dropped = drop_expired_records(state_with_new_records.latestTimestamp, state_with_new_records.currentTransactions)

        # Check if it's time to emit a record for this customer
        if F.current_timestamp() >= state_with_records_dropped.latestTimestamp + F.interval(1, "minute"):
            output = TransactionCount(customer_id, len(state_with_records_dropped.currentTransactions), F.current_timestamp(), False)
            transaction_counts.append(output)

        state.update(state_with_records_dropped)
        state.setTimeoutTimestamp(state_with_records_dropped.latestTimestamp + F.interval(30, "seconds"))
    else:
        prev_state = state.get()
        new_timestamp = F.current_timestamp()
        
        # Check if it's time to emit a record for this customer
        if new_timestamp >= prev_state.latestTimestamp + F.interval(1, "minute"):
            state_with_records_dropped = drop_expired_records(new_timestamp, prev_state.currentTransactions)
            output = TransactionCount(customer_id, len(state_with_records_dropped.currentTransactions), new_timestamp, True)
            transaction_counts.append(output)

        state.update(state_with_records_dropped)
        state.setTimeoutTimestamp(state_with_records_dropped.latestTimestamp + F.interval(30, "seconds"))

    return transaction_counts

schema = StructType([
    StructField("TransactionTimestamp", StringType(), True),
    StructField("CustomerID", IntgerType(), True),
    StructField("Amount", DoubleType(), True),
    StructField("Product", StringType(), True)
])

input_df = spark.readStream \
    .format("delta") \
    .schema(schema) \
    .table(input_table) \
    .selectExpr("CustomerID", "cast(TransactionTimestamp as timestamp) TransactionTimestamp") \
    .as[InputRow]
