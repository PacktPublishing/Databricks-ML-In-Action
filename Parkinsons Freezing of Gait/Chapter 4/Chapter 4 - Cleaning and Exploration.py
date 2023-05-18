# Databricks notebook source
# MAGIC %run ./setup

# COMMAND ----------

#import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SHOW TABLES LIKE 'park*'

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data File and Field Descriptions from Kaggle
# MAGIC
# MAGIC train/ Folder containing the data series in the training set within three subfolders: tdcsfog/, defog/, and notype/. Series in the notype folder are from the defog dataset but lack event-type annotations. The fields present in these series vary by folder.
# MAGIC * **Time** An integer timestep. Series from the tdcsfog dataset are recorded at 128Hz (128 timesteps per second), while series from the defog and daily series are recorded at 100Hz (100 timesteps per second).
# MAGIC * **AccV, AccML, and AccAP** Acceleration from a lower-back sensor on three axes: V - vertical, ML - mediolateral, AP - anteroposterior. Data is in units of m/s^2 for tdcsfog/ and g for defog/ and notype/.
# MAGIC * **StartHesitation**, Turn, Walking Indicator variables for the occurrence of each of the event types.
# MAGIC * **Event** Indicator variable for the occurrence of any FOG-type event. Present only in the notype series, which lack type-level annotations.
# MAGIC * **Valid** There were cases during the video annotation that were hard for the annotator to decide if there was an Akinetic (i.e., essentially no movement) FoG or the subject stopped voluntarily. Only event annotations where the series is marked true should be considered as unambiguous.
# MAGIC * **Task** Series were only annotated where this value is true. Portions marked false should be considered unannotated.
# MAGIC
# MAGIC Note that the Valid and Task fields are only present in the defog dataset. They are not relevant for the tdcsfog data.

# COMMAND ----------

from pyspark.sql.functions import sum
import matplotlib.pyplot as plt

df = spark.read.table("parkinsons_train_tdcsfog")
display(df)
#plot(df.select(sum("StartHesitation"),sum("Turn"), sum("Walking")).collect())


# COMMAND ----------

# MAGIC %md
# MAGIC Based of the basic data profile, we see that the our indicator variables are integers rather than boolean.

# COMMAND ----------

df = spark.read.table("parkinsons_train_tdcsfog")


# COMMAND ----------

# MAGIC %sql
# MAGIC -- The tDCS FOG (tdcsfog) dataset, comprising data series collected in the lab, as subjects completed a FOG-provoking protocol.
# MAGIC SELECT *, boolean(StartHesitation) FROM parkinsons_train_tdcsfog

# COMMAND ----------

df

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC *, 
# MAGIC CASE 
# MAGIC WHEN (Task == true AND Valid == true AND (StartHesitation + Turn + Walking) > 0)
# MAGIC THEN 1 ELSE 0 END AS EventInProgress
# MAGIC FROM lakehouse_in_action.parkinsons_train_defog

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 

df = sql('select *,  from lakehouse_in_action.parkinsons_train_defog where Valid == TRUE AND Task == TRUE AND (StartHesitation + Turn + Walking) > 0')

df_index = df.select("*").withColumn("id", monotonically_increasing_id())
display(df_index)
#df_index.write.mode("overwrite").saveAsTable("unambiguous_indexed")

# COMMAND ----------

# MAGIC %md We could use bamboolib here, or we could use the pandas profiler

# COMMAND ----------


