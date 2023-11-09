# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 6: Searching for Signal
# MAGIC
# MAGIC ## ASL Fingerspelling - asl-fingerspelling-recognition-w-tensorflow
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)

# COMMAND ----------

# MAGIC %md
# MAGIC # Google - American Sign Language Fingerspelling Recognition with TensorFlow
# MAGIC
# MAGIC This notebook walks you through how to train a Transformer model using TensorFlow on the Google - American Sign Language Fingerspelling Recognition dataset made available for this competition.
# MAGIC
# MAGIC The objective of the model is to predict and translate American Sign Language (ASL) fingerspelling from a set of video frames into text(`phrase`).
# MAGIC
# MAGIC In this notebook you will learn:
# MAGIC
# MAGIC - How to load the data
# MAGIC - Convert the data to tfrecords to make it faster to re-traing the model
# MAGIC - Train a transformer models on the data
# MAGIC - Convert the model to TFLite
# MAGIC - Create a submission

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md
# MAGIC # Import the libraries

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

from skimage.transform import resize
from mediapipe.framework.formats import landmark_pb2
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from matplotlib import animation, rc

# COMMAND ----------

print("TensorFlow v" + tf.__version__)
print("Mediapipe v" + mediapipe.__version__)
print("Pandas v" + pd.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the Dataset

# COMMAND ----------

dataset_df = spark.table("training_metadata").toPandas()
print("Full train dataset shape is {}".format(dataset_df.shape))

# COMMAND ----------

# MAGIC %md
# MAGIC The data is composed of 5 columns and 67208 entries. We can see all 5 dimensions of our dataset by printing out the first 5 entries using the following code:

# COMMAND ----------

display(dataset_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocess the data
# MAGIC
# MAGIC For convenience and efficiency, we will rearrange the data so that each parquet file contains the landmark data along with the phrase it represents. This way we don't have to switch between train.csv and its parquet file. 
# MAGIC
# MAGIC We will save the new data in the [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format. `TFRecord` format is a simple format for storing a sequence of binary records. Storing and loading the data using `TFRecord` is much more efficient and faster.
# MAGIC
# MAGIC Reference:
# MAGIC
# MAGIC https://www.kaggle.com/code/irohith/aslfr-preprocess-dataset
# MAGIC
# MAGIC https://www.kaggle.com/code/shlomoron/aslfr-parquets-to-tfrecords-cleaned

# COMMAND ----------

# MAGIC %md
# MAGIC # Fetch the pose landmark coordinates related to hand movement.

# COMMAND ----------

# Pose coordinates for hand movement.
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

# COMMAND ----------

# MAGIC %md
# MAGIC # Create x,y,z label names from coordinates

# COMMAND ----------

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

# COMMAND ----------

# MAGIC %md
# MAGIC Create feature columns from the extracted coordinates.

# COMMAND ----------

FEATURE_COLUMNS = X + Y + Z

# COMMAND ----------

# MAGIC %md
# MAGIC Store ids of each coordinate labels to lists

# COMMAND ----------

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocess and write the dataset as TFRecords

# COMMAND ----------

#dbutils.fs.mkdirs("/Volumes/lakehouse_in_action/asl_fingerspelling/asl_volume/preprocessed")


# COMMAND ----------

# Set length of frames to 128
FRAME_LEN = 128

# # Loop through each file_id
# for file_id in tqdm(dataset_df.file_id.unique()):
#     # Parquet file name
#     pq_file = f"s3://one-env/lakehouse_ml_in_action/asl-fingerspelling/train_landmarks/{file_id}.parquet"
#     # Filter train.csv and fetch entries only for the relevant file_id
#     file_df = dataset_df.loc[dataset_df["file_id"] == file_id]
#     # Fetch the parquet file
#     parquet_df = pq.read_table(f"s3://one-env/lakehouse_ml_in_action/asl-fingerspelling/train_landmarks/{str(file_id)}.parquet",
#                               columns=['sequence_id'] + FEATURE_COLUMNS).to_pandas()
#     # File name for the updated data
#     tf_file = f"/Volumes/lakehouse_in_action/asl_fingerspelling/asl_volume/preprocessed/{file_id}.tfrecord"
#     parquet_numpy = parquet_df.to_numpy()
#     # Initialize the pointer to write the output of 
#     # each `for loop` below as a sequence into the file.
#     with tf.io.TFRecordWriter(tf_file) as file_writer:
#         # Loop through each sequence in file.
#         for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
#             # Fetch sequence data
#             frames = parquet_numpy[parquet_df.index == seq_id]
            
#             # Calculate the number of NaN values in each hand landmark
#             r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis = 1) == 0)
#             l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis = 1) == 0)
#             no_nan = max(r_nonan, l_nonan)
            
#             if 2*len(phrase)<no_nan:
#                 features = {FEATURE_COLUMNS[i]: tf.train.Feature(
#                     float_list=tf.train.FloatList(value=frames[:, i])) for i in range(len(FEATURE_COLUMNS))}
#                 features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
#                 record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
#                 file_writer.write(record_bytes)

# COMMAND ----------

# MAGIC %md
# MAGIC  Load the preprocessed data

# COMMAND ----------

# MAGIC %md
# MAGIC # Get the saved TFRecord files into a list

# COMMAND ----------

tf_records = dataset_df.file_id.map(lambda x: f'{volume_data_path}preprocessed/{x}.tfrecord').unique()
print(f"List of {len(tf_records)} TFRecord files.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load character_to_prediction json file

# COMMAND ----------

# MAGIC %md
# MAGIC This json file contains a character and its value. We will add three new characters, "<" and ">" to mark the start and end of each phrase, and "P" for padding.

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

# Reference: https://www.kaggle.com/code/irohith/aslfr-transformer/notebook

# Function to resize and add padding.
def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN-tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x

# Detect the dominant hand from the number of NaN values.
# Dominant hand will have less NaN values since it is in frame moving.
def pre_process(x):
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)
    
    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)
    
    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)
    
    # For dominant hand
    if rnans > lnans:
        hand = lhand
        pose = lpose
        
        hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
        hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
        hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
        hand = tf.concat([1-hand_x, hand_y, hand_z], axis=1)
        
        pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
        pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
        pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
        pose = tf.concat([1-pose_x, pose_y, pose_z], axis=1)
    else:
        hand = rhand
        pose = rpose
    
    hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
    hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
    hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)
    
    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    hand = (hand - mean) / std

    pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
    pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
    pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
    pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)
    
    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)
    
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    return x

# COMMAND ----------

# MAGIC %md
# MAGIC # Create function to parse data from TFRecord format
# MAGIC
# MAGIC This function will read the `TFRecord` data and convert it to Tensors.

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

# MAGIC %md
# MAGIC # Create function to convert the data 
# MAGIC
# MAGIC This function transposes and applies masks to the landmark coordinates. It also vectorizes the phrase corresponding to the landmarks using `character_to_prediction_index.json`.

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

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC Use the functions we defined above to create the final dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC # Train and validation split/Create the final datasets

# COMMAND ----------

batch_size = 64
train_len = int(0.8 * len(tf_records))

train_ds = tf.data.TFRecordDataset(tf_records[:train_len]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
valid_ds = tf.data.TFRecordDataset(tf_records[train_len:]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create the Transformer model
# MAGIC
# MAGIC We will use a **[Transformer](https://en.wikipedia.org/wiki/Transformer_machine_learning_model)** to train a model for this dataset. **Transformers** are designed to process sequential input data. The model we are going to design is similar to the one used in the [Automatic Speech Recognition with Transformer](https://keras.io/examples/audio/transformer_asr/) tutorial for **Keras**. We will finetune only a small part of the model since we can treat the ASL Fingerspelling recognition problem similar to speech recognition. In both cases, we have to predict a sentence from a sequence of data.
# MAGIC
# MAGIC ![picnogrid.gif](attachment:9c4fc53c-6b08-4404-9369-34f391fb78d6.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC # Define the Transformer Input Layers
# MAGIC
# MAGIC When processing landmark coordinate features for the encoder, we apply convolutional layers to downsample them and process local relationships. 
# MAGIC
# MAGIC We sum position embeddings and token embeddings when processing past target tokens for the decoder.

# COMMAND ----------

class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)

# COMMAND ----------

# MAGIC %md
# MAGIC # Encoder layer for Transformer

# COMMAND ----------

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# COMMAND ----------

# MAGIC %md
# MAGIC # Decoder layer for Transformer

# COMMAND ----------

# Customized to add `training` variable
# Reference: https://www.kaggle.com/code/shlomoron/aslfr-a-simple-transformer/notebook

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target, training):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training = training))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out, training = training) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training = training))
        return ffn_out_norm

# COMMAND ----------

# MAGIC %md
# MAGIC # Complete the Transformer model
# MAGIC
# MAGIC This model takes landmark coordinates as inputs and predicts a sequence of characters. The target character sequence, which has been shifted to the left is provided as the input to the decoder during training. The decoder employs its own past predictions during inference to forecast the next token.
# MAGIC
# MAGIC The **Levenshtein Distance** between sequences is used as the accuracy metric since the evaluation metric for this contest is the **Normalized Total Levenshtein Distance**.

# COMMAND ----------

# Customized to add edit_dist metric and training variable.
# Reference:
# https://www.kaggle.com/code/irohith/aslfr-transformer/notebook
# https://www.kaggle.com/code/shlomoron/aslfr-a-simple-transformer/notebook

class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=60,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.Mean(name="edit_dist")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target, training):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y, training)
        return y

    def call(self, inputs, training):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source, training)
        y = self.decode(x, target, training)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch[0]
        target = batch[1]

        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, pad_token_idx))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Computes the Levenshtein distance between sequences since the evaluation
        # metric for this contest is the normalized total levenshtein distance.
        edit_dist = tf.edit_distance(tf.sparse.from_dense(target), 
                                     tf.sparse.from_dense(tf.cast(tf.argmax(preds, axis=1), tf.int32)))
        edit_dist = tf.reduce_mean(edit_dist)
        self.acc_metric.update_state(edit_dist)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result(), "edit_dist": self.acc_metric.result()}

    def test_step(self, batch):        
        source = batch[0]
        target = batch[1]

        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, pad_token_idx))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        # Computes the Levenshtein distance between sequences since the evaluation
        # metric for this contest is the normalized total levenshtein distance.
        edit_dist = tf.edit_distance(tf.sparse.from_dense(target), 
                                     tf.sparse.from_dense(tf.cast(tf.argmax(preds, axis=1), tf.int32)))
        edit_dist = tf.reduce_mean(edit_dist)
        self.acc_metric.update_state(edit_dist)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result(), "edit_dist": self.acc_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source, training = False)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input, training = False)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

# COMMAND ----------

# MAGIC %md
# MAGIC The following callback function is used to display predictions.

# COMMAND ----------

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=60, target_end_token_idx=61
    ):
        """Displays a batch of outputs after every 4 epoch

        Args:
            batch: A test batch
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 != 0:
            return
        source = self.batch[0]
        target = self.batch[1].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC # Train the Transformer model

# COMMAND ----------

# Transformer variables are customized from original keras tutorial to suit this dataset.
# Reference: https://www.kaggle.com/code/shlomoron/aslfr-a-simple-transformer/notebook

batch = next(iter(valid_ds))

# The vocabulary to convert predicted indices into characters
idx_to_char = list(char_to_num.keys())
display_cb = DisplayOutputs(
    batch, idx_to_char, target_start_token_idx=char_to_num['<'], target_end_token_idx=char_to_num['>']
)  # set the arguments as per vocabulary index for '<' and '>'

model = Transformer(
    num_hid=200,
    num_head=4,
    num_feed_forward=400,
    source_maxlen = FRAME_LEN,
    target_maxlen=64,
    num_layers_enc=2,
    num_layers_dec=1,
    num_classes=62
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1,
)


optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss=loss_fn)

history = model.fit(train_ds, validation_data=valid_ds, callbacks=[display_cb], epochs=13)

# COMMAND ----------

# MAGIC %md
# MAGIC # Plot training loss and validation loss

# COMMAND ----------

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'val_loss'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Submission
# MAGIC
# MAGIC Refrence: https://www.kaggle.com/code/shlomoron/aslfr-a-simple-transformer/notebook

# COMMAND ----------

# MAGIC %md
# MAGIC # Create TFLite model

# COMMAND ----------

 class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.target_start_token_idx = start_token_idx
        self.target_end_token_idx = end_token_idx
        # Load the feature generation and main models
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(FEATURE_COLUMNS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        # Preprocess Data
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(FEATURE_COLUMNS))), lambda: tf.identity(x))
        x = x[0]
        x = pre_process(x)
        x = x[None]
        x = self.model.generate(x, self.target_start_token_idx)
        x = x[0]
        idx = tf.argmax(tf.cast(tf.equal(x, self.target_end_token_idx), tf.int32))
        idx = tf.where(tf.math.less(idx, 1), tf.constant(2, dtype=tf.int64), idx)
        x = x[1:idx]
        x = tf.one_hot(x, 59)
        return {'outputs': x}
    
tflitemodel_base = TFLiteModel(model)

# COMMAND ----------

# dbutils.fs.mkdirs(f"{volume_data_path}models/")
# model.save_weights(f"{volume_data_path}models/model.h5")

# COMMAND ----------

keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel_base)
keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]#, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = keras_model_converter.convert()
with open(f'{volume_data_path}models/model.tflite', 'wb') as f:
    f.write(tflite_model)
    
infargs = {"selected_columns" : FEATURE_COLUMNS}

with open(f'{volume_data_path}models/inference_args.json', "w") as json_file:
    json.dump(infargs, json_file)

# COMMAND ----------

interpreter = tf.lite.Interpreter(f"{volume_data_path}models/model.tflite")

REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

rev_character_map = {j:i for i,j in char_to_num.items()}

found_signatures = list(interpreter.get_signature_list().keys())

if REQUIRED_SIGNATURE not in found_signatures:
    raise KernelEvalException('Required input signature not found.')

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

from mlflow.models.signature import infer_signature
# Create a model signature using the tensor input to store in the MLflow model registry
signature = infer_signature(np.expand_dims(batch[0][0], axis=0), prediction_fn(inputs=batch[0][0]))
# Let's check out how it looks
print(signature)

# COMMAND ----------

np.expand_dims(batch[0][0], axis=0)

# COMMAND ----------

from mlflow.types.schema import Schema, TensorSpec
from mlflow.models import ModelSignature, infer_signature
import mlflow

mlflow.set_registry_uri("databricks-uc")

# # Option 1: Manually construct the signature object
# input_schema = Schema(
#     [
#       TensorSpec((-1,len(FEATURE_COLUMNS)), type=np.dtype(np.float32)),
#     ]
# )
# output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
# signature = ModelSignature(inputs=input_schema, outputs=output_schema)

MODEL_NAME = "lakehouse_in_action.asl_fingerspelling.phrase_predictions"

mlflow.tensorflow.log_model(
tflitemodel_base,
artifact_path="model",
signature=signature,
registered_model_name=MODEL_NAME
)
