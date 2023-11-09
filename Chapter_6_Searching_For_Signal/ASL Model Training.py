# Databricks notebook source
# MAGIC %md
# MAGIC # ASL Fingerspelling
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
# MAGIC
# MAGIC ##Run Setup

# COMMAND ----------

#dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.pandas as psp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage.transform import resize
from mediapipe.framework.formats import landmark_pb2
import matplotlib
import matplotlib.pyplot as plt
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

FEATURE_COLS = (
    [f"x_hand_{i}" for i in range(21)]
    + [f"y_hand_{i}" for i in range(21)]
    + [f"z_hand_{i}" for i in range(21)]
    + [f"x_pose_{i}" for i in range(5)]
    + [f"y_pose_{i}" for i in range(5)]
    + [f"z_pose_{i}" for i in range(5)]
)

HAND_IDX = [i for i, col in enumerate(FEATURE_COLS) if "hand" in col]
POSE_IDX = [i for i, col in enumerate(FEATURE_COLS) if "pose" in col]

# COMMAND ----------

tf_file_ids = spark.table("cleaned_training_metadata").select("file_id").distinct().toPandas()
tf_records = tf_file_ids.file_id.map(lambda x: f'/Volumes/lakehouse_in_action/asl_fingerspelling/asl_volume/tfrecords/{x}.tfrecord')

print(f"List of {len(tf_records)} TFRecord files.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Build a library of functions
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load character_to_prediction json file
# MAGIC
# MAGIC This json file contains a character and its value. We will add three new characters, "<" and ">" to mark the start and end of each phrase, and "P" for padding.

# COMMAND ----------

char_to_num = spark.table("character_to_prediction_index")

# Add pad_token, start pointer and end pointer to the dict
update_dict = {}
pad_token = 'P'
start_token = '<'
end_token = '>'
pad_token_idx = 59
start_token_idx = 60
end_token_idx = 61

update_dict[pad_token] = pad_token_idx
update_dict[start_token] = start_token_idx
update_dict[end_token] = end_token_idx
update_df = spark.createDataFrame([(key,value) for key, value in update_dict.items()], schema=["char","pred_index"])

char_to_num = char_to_num.unionAll(update_df)
char_to_num = char_to_num.toPandas()
char_to_num = dict(zip(char_to_num.char, char_to_num.pred_index))

# COMMAND ----------

# Reference: https://www.kaggle.com/code/irohith/aslfr-transformer/notebook

# Set length of frames to 128
FRAME_LEN = 128

# Function to resize and add padding.
def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN-tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x


def pre_process(frames):
  hand_x = frames[:, (0 * len(HAND_IDX)//3) : (1 * len(HAND_IDX)//3)]
  hand_y = frames[:, (1 * len(HAND_IDX)//3) : (2 * len(HAND_IDX)//3)]
  hand_z = frames[:, (2 * len(HAND_IDX)//3) : (3 * len(HAND_IDX)//3)]
  hand = tf.concat(
      [hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]],
      axis=-1,
  )

  mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
  std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
  hand = (hand - mean) / std

  hands_end = len(HAND_IDX)
  pose_x = frames[:, hands_end + (0 * len(POSE_IDX)//3) : hands_end + (1 * len(POSE_IDX)//3)]
  pose_y = frames[:, hands_end + (1 * len(POSE_IDX)//3) : hands_end + (2 * len(POSE_IDX)//3)]
  pose_z = frames[:, hands_end + (2 * len(POSE_IDX)//3) : hands_end + (3 * len(POSE_IDX)//3)]
  pose = tf.concat(
      [pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]],
      axis=-1,
  )

  frames = tf.concat([hand, pose], axis=1)
  frames = resize_pad(frames)

  frames = tf.where(tf.math.is_nan(frames), tf.zeros_like(frames), frames)
  frames = tf.reshape(frames, (FRAME_LEN, len(HAND_IDX) + len(POSE_IDX)))
  return frames

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create function to parse data from TFRecord format
# MAGIC
# MAGIC This function will read the `TFRecord` data and convert it to Tensors.

# COMMAND ----------

def decode_fn(record_bytes):
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]
    landmarks = ([tf.sparse.to_dense(features[COL]) for COL in FEATURE_COLS])
    # Transpose to maintain the original shape of landmarks data.
    landmarks = tf.transpose(landmarks)
    
    return landmarks, phrase

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create function to convert the data 
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
# MAGIC #Configure MLflow client to access models in Unity Catalog
# MAGIC
# MAGIC By default, the MLflow Python client creates models in the workspace model registry on Databricks. To upgrade to models in Unity Catalog, configure the client to access models in Unity Catalog:

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC # Train and validation split/Create the final datasets

# COMMAND ----------

batch_size = 64
train_len = int(0.8 * len(tf_records))

train_ds = tf.data.TFRecordDataset(tf_records[:train_len]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
valid_ds = tf.data.TFRecordDataset(tf_records[train_len:]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
ex_ds = tf.data.TFRecordDataset(tf_records[:2]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()


# COMMAND ----------

ex_ds = tf.data.TFRecordDataset(tf_records[:10]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

display(ex_ds.take(3))

# COMMAND ----------

# MAGIC %md
# MAGIC # Create the Transformer model
# MAGIC
# MAGIC We will use a **[Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))** to train a model for this dataset. **Transformers** are designed to process sequential input data. The model we are going to design is similar to the one used in the [Automatic Speech Recognition with Transformer](https://keras.io/examples/audio/transformer_asr/) tutorial for **Keras**. We will finetune only a small part of the model since we can treat the ASL Fingerspelling recognition problem similar to speech recognition. In both cases, we have to predict a sentence from a sequence of data.
# MAGIC
# MAGIC ![picnogrid.gif](attachment:9c4fc53c-6b08-4404-9369-34f391fb78d6.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Transformer Input Layers
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
# MAGIC # Train and Register the Transformer model

# COMMAND ----------

# Transformer variables are customized from the reference notebook to include the MLflow model registry
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

history = model.fit(train_ds, validation_data=valid_ds, callbacks=[display_cb], epochs=2)


# model,history = train_and_register_keras_model(train_set=train_ds, validation_set=valid_ds,example_set=ex_ds)


# COMMAND ----------

# MAGIC %md
# MAGIC # Plot training loss and validation loss

# COMMAND ----------

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'val_loss'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Create TFLite model
# MAGIC
# MAGIC Refrence: https://www.kaggle.com/code/shlomoron/aslfr-a-simple-transformer/notebook

# COMMAND ----------

 class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.target_start_token_idx = start_token_idx
        self.target_end_token_idx = end_token_idx
        # Load the feature generation and main models
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, len(FEATURE_COLS)], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        # Preprocess Data
        x = tf.cast(inputs, tf.float32)
        x = x[None]
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, len(FEATURE_COLS))), lambda: tf.identity(x))
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

from mlflow.models import ModelSignature, infer_signature

signature = infer_signature(train_ds, tflitemodel_base(train_ds))

# COMMAND ----------

keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel_base)
keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]#, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = keras_model_converter.convert()

# COMMAND ----------

interpreter = tf.lite.Interpreter("model.tflite")


# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(valid_ds, model.predict(valid_ds))


# COMMAND ----------

from mlflow.types.schema import Schema, TensorSpec

# Option 1: Manually construct the signature object
input_schema = Schema(
    [
      TensorSpec(shape=[None, len(FEATURE_COLS)], type=np.dtype(np.float32), name='inputs'),
    ]
)
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

MODEL_NAME = "lakehouse_in_action.asl_fingerspelling.phrase_predictions"

mlflow.tensorflow.log_model(
tflitemodel_base,
artifact_path="model",
signature=signature,
registered_model_name=MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Use the MLflow Client to access and alias your model

# COMMAND ----------

def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

client = MlflowClient()
latest_version = client.get_latest_model_version(MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "Baseline TF", latest_version)

# COMMAND ----------

keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflitemodel_base)
keras_model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]#, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = keras_model_converter.convert()
with open('"/Volumes/lakehouse_in_action/asl_fingerspelling/asl_volume/models/model.tflite', 'wb') as f:
    f.write(tflite_model)
    
infargs = {"selected_columns" : FEATURE_COLS}

with open('inference_args.json', "w") as json_file:
    json.dump(infargs, json_file)

# COMMAND ----------

interpreter = tf.lite.Interpreter("model.tflite")

REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

with open ("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}

found_signatures = list(interpreter.get_signature_list().keys())

if REQUIRED_SIGNATURE not in found_signatures:
    raise KernelEvalException('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")
output = prediction_fn(inputs=batch[0][0])
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
print(prediction_str)
