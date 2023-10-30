# Databricks notebook source
# MAGIC %run ./../../global-setup $project_name=parkinsons-freezing_gait_prediction $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Try out the Databricks Assistant.
# MAGIC Lets check that Subject & Visit make up the primary key for the Subject table
# MAGIC
# MAGIC Let's ask "How many pairs of Subject, Visit are in parkinsons_subjects?"

# COMMAND ----------

# DBTITLE 1,This is what I asked for, but not what I wanted.
# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT Subject, Visit) AS pairs_count
# MAGIC FROM parkinsons_subjects

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try again with better wording "What are the Subject, Visit combinations and how many are there of each?"

# COMMAND ----------

# DBTITLE 1,This is the answer we were looking for.
# MAGIC %sql
# MAGIC SELECT Subject, Visit, count(*) as count FROM parkinsons_subjects GROUP BY Subject, Visit

# COMMAND ----------

# MAGIC %md
# MAGIC Asking the same question for the `tdcsfog_metadata` table shows Subject, Visit are not a primary key. In the documentation, it explains that Visit in the `defog_metadata` and the `tdcsfog_metadata` are different column meanings with the same name. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   Subject,
# MAGIC   Visit,
# MAGIC   COUNT(*) AS combo_count
# MAGIC FROM
# MAGIC   parkinsons_tdcsfog_metadata
# MAGIC GROUP BY
# MAGIC   Subject,
# MAGIC   Visit
# MAGIC ORDER BY
# MAGIC   Subject

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Subject, Visit, COUNT(*) AS count
# MAGIC FROM parkinsons_tdcsfog_metadata
# MAGIC GROUP BY Subject, Visit
