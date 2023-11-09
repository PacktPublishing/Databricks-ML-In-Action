# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 7: Production
# MAGIC
# MAGIC ##  ASL Fingerspelling - ASL Model Serving
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)

# COMMAND ----------

# MAGIC %md ## Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import mediapipe
import matplotlib
import matplotlib.pyplot as plt
import random
import json

import os
import requests
import numpy as np
import pandas as pd
import json

from skimage.transform import resize
from mediapipe.framework.formats import landmark_pb2
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from matplotlib import animation, rc

# COMMAND ----------

shard_url = '<put-shard-url-here (e.g. https://company.cloud.databricks.com)>'
model_name = 'lakehouse_in_action.asl_fingerspelling.phrase_predictions'

# COMMAND ----------

# Set length of frames to 128
FRAME_LEN = 128

serving_demo_df = spark.table("supplemental_metadata").toPandas()
print("The supplemental dataset shape is {}".format(serving_demo_df.shape))

# COMMAND ----------

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

 
def process_input(dataset):
  if isinstance(dataset, pd.DataFrame):
    return {'dataframe_split': dataset.to_dict(orient='split')}
  elif isinstance(dataset, str):
    return dataset
  else:
    return create_tf_serving_json(dataset)
 
 
def score_model(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/phrase-prediction/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 
'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

tf_records = dataset_df.file_id.map(lambda x: f'{volume_data_path}preprocessed/{x}.tfrecord').unique()
print(f"List of {len(tf_records)} TFRecord files.")

# COMMAND ----------

with open (f"{volume_data_path}/character_to_prediction_index.json", "r") as f:
    char_to_num = json.load(f)

# Add pad_token, start pointer and end pointer to the dict
pad_token = 'P'
start_token = '<'
end_token = '>'
pad_token_idx = 59
start_token_idx = 60
end_token_idx = 61

char_to_num[pad_token] = pad_token_idx
char_to_num[start_token] = start_token_idx
char_to_num[end_token] = end_token_idx
num_to_char = {j:i for i,j in char_to_num.items()}

# COMMAND ----------

def decode_fn(record_bytes):
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLUMNS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]
    landmarks = ([tf.sparse.to_dense(features[COL]) for COL in FEATURE_COLUMNS])
    # Transpose to maintain the original shape of landmarks data.
    landmarks = tf.transpose(landmarks)
    
    return landmarks, phrase

# COMMAND ----------

table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=list(char_to_num.keys()),
        values=list(char_to_num.values()),
    ),
    default_value=tf.constant(-1),
    name="class_weight"
)

def convert_fn(landmarks, phrase):
    # Add start and end pointers to phrase.
    phrase = start_token + phrase + end_token
    phrase = tf.strings.bytes_split(phrase)
    phrase = table.lookup(phrase)
    # Vectorize and add padding.
    phrase = tf.pad(phrase, paddings=[[0, 64 - tf.shape(phrase)[0]]], mode = 'CONSTANT',
                    constant_values = pad_token_idx)
    # Apply pre_process function to the landmarks.
    return pre_process(landmarks), phrase

# COMMAND ----------

batch_size = 64
train_len = int(0.8 * len(tf_records))

train_ds = tf.data.TFRecordDataset(tf_records[:train_len]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
valid_ds = tf.data.TFRecordDataset(tf_records[train_len:]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

# COMMAND ----------

rev_character_map = {j:i for i,j in char_to_num.items()}

prediction_fn = interpreter.get_signature_runner("serving_default")
output = prediction_fn(inputs=batch[0][0])
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
target = batch[1][0].numpy()
target_text = "".join([idx_to_char[_] for _ in target])
print(batch[0][0])
print(prediction_str)
print(batch[1][0])
print(target_text)

# COMMAND ----------

import mlflow
path = mlflow.artifacts.download_artifacts(f'models:/{model_name}/1')
model = mlflow.pyfunc.load_model(f'models:/{model_name}/1')
input_example = model.metadata.load_input_example(path)

# COMMAND ----------

score_model(input_example)

# COMMAND ----------


