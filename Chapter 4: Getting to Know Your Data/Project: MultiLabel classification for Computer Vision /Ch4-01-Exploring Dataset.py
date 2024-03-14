# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Chapter 4: Getting to Know Your Data
# MAGIC
# MAGIC ## Intel Multilable Image Classification - Explore your dataset.

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

from mlia_utils.cv_clf_funcs import *

train_df = (spark.read.format("delta").load(train_delta_path))
print(idx_class(train_df))
display(train_df)

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

display_image(f"{train_dir}/forest/17856.jpg")
display_image(f"{train_dir}/street/15478.jpg")

# COMMAND ----------

proportion_labels(labels_dict_train)

# COMMAND ----------

proportion_labels(labels_dict_valid)

# COMMAND ----------


