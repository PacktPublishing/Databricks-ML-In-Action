# Databricks notebook source
# MAGIC %run ./../setup

# COMMAND ----------

from pyspark.sql.functions import lag, col, stddev, avg
from pyspark.sql.window import Window
#from outliers import smirnov_grubbs as grubbs


# COMMAND ----------

# Assuming the data has a "timestamp" column and a "value" column

# Perform seasonal decomposition
seasonality_window = 10  # Adjust the window size as needed
w = Window.orderBy(col("timestamp").cast("long")).rowsBetween(-seasonality_window, seasonality_window)
data = data.withColumn("value_avg", avg(col("value")).over(w))
data = data.withColumn("value_seasonal", col("value") - col("value_avg"))

# Apply the Seasonal Hybrid ESD algorithm for event detection
# Adjust the parameters as needed
alpha = 0.05
max_outliers = 5
seasonal_period = 24  # Assuming hourly data, adjust as needed

outliers = []
while len(outliers) < max_outliers:
    # Decompose the residual values
    decomposition = seasonal_decompose(data.select("value_seasonal").rdd.map(lambda x: x[0]).collect(),
                                       period=seasonal_period, two_sided=False)

    # Calculate the test statistics using the Grubbs' test
    test_statistic = grubbs.max_test_statistic(decomposition.resid, alpha)

    # Find the index of the largest outlier in the residual values
    outlier_index = decomposition.resid.tolist().index(max(decomposition.resid, key=abs))

    # Add the outlier to the list
    outliers.append(data.select("timestamp").collect()[outlier_index][0])

    # Remove the outlier from the data
    data = data.filter(col("timestamp") != outliers[-1])

# Print the detected event timestamps
for outlier in outliers:
    print(outlier)

# Stop the SparkSession
spark.stop()


# COMMAND ----------

# MAGIC %scala
# MAGIC from pyspark.sql import SparkSession
# MAGIC from pyspark.sql.functions import col, stddev, avg
# MAGIC
# MAGIC # Create a SparkSession
# MAGIC spark = SparkSession.builder.appName("TimeSeriesAnomalyDetection").getOrCreate()
# MAGIC
# MAGIC # Read the time series data into a DataFrame
# MAGIC data = spark.read.csv("path/to/timeseries_data.csv", header=True, inferSchema=True)
# MAGIC
# MAGIC # Assuming the data has a "timestamp" column and a "value" column
# MAGIC
# MAGIC # Calculate the mean and standard deviation of the values
# MAGIC mean_value = data.select(avg(col("value"))).first()[0]
# MAGIC stddev_value = data.select(stddev(col("value"))).first()[0]
# MAGIC
# MAGIC # Define the threshold for anomaly detection
# MAGIC threshold = 3  # Adjust the threshold as needed
# MAGIC
# MAGIC # Detect anomalies based on the Z-Score
# MAGIC data = data.withColumn("z_score", (col("value") - mean_value) / stddev_value)
# MAGIC anomalies = data.filter(col("z_score").abs() > threshold)
# MAGIC
# MAGIC # Print the detected anomaly timestamps
# MAGIC anomaly_timestamps = anomalies.select("timestamp").collect()
# MAGIC for row in anomaly_timestamps:
# MAGIC     print(row[0])
# MAGIC
# MAGIC # Stop the SparkSession
# MAGIC spark.stop()
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


