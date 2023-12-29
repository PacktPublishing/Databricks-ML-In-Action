# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Chapter 4
# MAGIC
# MAGIC ## Intel Multilable Image Classification - Explore your dataset.

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM delta.`/Volumes/$catalog/$schema/intel_image_clf/valid_imgs_main.delta`

# COMMAND ----------

from mlia_utils.cv_clf_funcs import *

train_delta_path = f"/Volumes/{catalog}/{database_name}/intel_image_clf/train_imgs_main.delta"
val_delta_path = f"/Volumes/{catalog}/{database_name}/intel_image_clf/valid_imgs_main.delta"

train_df = (spark.read.format("delta").load(train_delta_path))
            
print(idx_class(train_df))

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

display_image(f"{train_dir}/forest/17856.jpg")
display_image(f"{train_dir}/street/15478.jpg")

# COMMAND ----------

proportion_labels(labels_dict_train)

# COMMAND ----------

proportion_labels(labels_dict_valid)
