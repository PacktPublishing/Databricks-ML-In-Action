# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Chapter 4

# COMMAND ----------

dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "")

# COMMAND ----------

import os 

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

MAIN_DIR_UC = f"/Volumes/{catalog_name}/{schema_name}/intel_image_clf/raw_images"
MAIN_DIR2Write = "/Volumes/{catalog_name}/{schema_name}/intel_image_clf/"
data_dir_Train = f"{MAIN_DIR_UC}/seg_train"
data_dir_Test = f"{MAIN_DIR_UC}/seg_test"
data_dir_pred = f"{MAIN_DIR_UC}/seg_pred/seg_pred"

train_dir = data_dir_Train + "/seg_train"
valid_dir = data_dir_Test + "/seg_test"
pred_files = [os.path.join(data_dir_pred, f) for f in os.listdir(data_dir_pred)]

labels_dict_train = {f"{f}":len(os.listdir(os.path.join(train_dir, f))) for f in os.listdir(train_dir)}
labels_dict_valid = {f"{f}":len(os.listdir(os.path.join(valid_dir, f))) for f in os.listdir(valid_dir)}

outcomes = os.listdir(train_dir)
print(outcomes)

# COMMAND ----------

def proportion_labels(labels_dict_name):  
  import numpy as np
  final_s = np.zeros(6)
  ticks = []
  idx_array = np.zeros(6)
  
  # plot the pie chart and bar graph of labels
  for idx,(ikey, ival) in enumerate(labels_dict_name.items()):
    print(ival,ikey, idx)
    final_s[idx] = ival
    idx_array[idx] = idx+1
    ticks.append(ikey)
    
  import matplotlib.pyplot as plt
  plt.figure(figsize=(20,9))

  plt.subplot(121)
  plt.bar(idx_array, final_s)
  plt.xticks(ticks=idx_array, labels=ticks, fontsize=12, fontweight='bold')
  plt.yticks(fontsize=12, fontweight='bold')
  plt.grid(visible=True)
  plt.title("Number of images per class", size=14, weight='bold')

  plt.subplot(122)
  plt.pie(final_s.ravel(),
          labels=ticks,
          autopct='%1.2f%%',
          textprops={'fontweight':'bold'}
          )
  plt.title("proportion of classes", size=14, weight='bold')

  plt.suptitle(f"Proportion of data", size=20, weight='bold')
  plt.show()

# COMMAND ----------

proportion_labels(labels_dict_train)

# COMMAND ----------

proportion_labels(labels_dict_valid)

# COMMAND ----------


