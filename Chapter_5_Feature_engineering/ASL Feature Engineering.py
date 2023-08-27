# Databricks notebook source
# MAGIC %md
# MAGIC # ASL Fingerspelling
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
# MAGIC
# MAGIC ##Run Setup

# COMMAND ----------

# MAGIC %run ../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess the data
# MAGIC
# MAGIC Again, the competition provides code specific to this dataset.
# MAGIC
# MAGIC For convenience and efficiency, we will rearrange the data so that each parquet file contains the landmark data along with the phrase it represents. This way we don't have to switch between train.csv and its parquet file. 
# MAGIC
# MAGIC We will save the new data in the [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format. `TFRecord` format is a simple format for storing a sequence of binary records. Storing and loading the data using `TFRecord` is much more efficient and faster.
# MAGIC
# MAGIC Reference:
# MAGIC
# MAGIC https://www.kaggle.com/code/gusthema/asl-fingerspelling-recognition-w-tensorflow/notebook
# MAGIC
# MAGIC https://www.kaggle.com/code/irohith/aslfr-preprocess-dataset
# MAGIC
# MAGIC https://www.kaggle.com/code/shlomoron/aslfr-parquets-to-tfrecords-cleaned

# COMMAND ----------

# Pose coordinates for hand movement.
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create x,y,z label names from coordinates

# COMMAND ----------

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

# COMMAND ----------

# MAGIC %md
# MAGIC Create feature column names from the extracted coordinates.

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
len(RHAND_IDX)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocess and write the dataset as TFRecords
# MAGIC
# MAGIC Using the extracted landmarks and phrases let us create new dataset files and write them as TFRecords.
# MAGIC
# MAGIC This takes around 10 minutes. After this, loading the dataset will be faster for any future experiments.

# COMMAND ----------

from pyspark.sql.functions import greatest, col, length
import pyspark.pandas as ps

# COMMAND ----------

# Set length of frames to 128
FRAME_LEN = 128

sequences = sql("SELECT DISTINCT sequence_id FROM train_landmarks")
df = spark.table("train_landmarks").select("sequence_id", *FEATURE_COLUMNS)
df = df.withColumn(
    "num_rh_null",
    sum(df[colm].isNull().cast("int") for colm in RHAND_IDX),
).withColumn(
    "num_lh_null",
    sum(df[colm].isNull().cast("int") for colm in LHAND_IDX),
)

# COMMAND ----------



# COMMAND ----------

rhdf = df.filter(col('num_rh_null')==0).groupBy('sequence_id').count().withColumnRenamed("count","rh_nn_rows")
lhdf = df.filter(col('num_lh_null')==0).groupBy('sequence_id').count().withColumnRenamed("count","lh_nn_rows")

mdf = spark.table("training_metadata").withColumn("phrase_length", length(col("phrase")))
mdf = mdf.join(lhdf, on='sequence_id', how='left').join(rhdf, on='sequence_id', how='left').fillna({'lh_nn_rows': 0,'rh_nn_rows': 0})
mdf = mdf.withColumn('max_nn_rows', greatest(col("lh_nn_rows"), col("rh_nn_rows")))

# COMMAND ----------



# COMMAND ----------

mdf.filter(2*col('phrase_length')<col('max_nn_rows')).createOrReplaceTempView("cleaned_training_metadata")

# COMMAND ----------



# COMMAND ----------

df.createOrReplaceTempView("feature_table")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM feature_table

# COMMAND ----------

import numpy as np

seq_id = '1983865658'
phrase = 'jeramy duran'
tdf= spark.table("train_landmarks").select("sequence_id", *FEATURE_COLUMNS).filter(col('sequence_id')==seq_id)
#mdf = spark.table("training_metadata").withColumn("phrase_length", length(col("phrase")))
#tdf = tdf.join(mdf, on ="sequence_id", how='inner')
pdf = tdf.toPandas()

parquet_numpy = tdf.pandas_api().to_numpy()


frames = parquet_numpy

# Calculate the number of NaN values in each hand landmark
r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
no_nan = max(r_nonan, l_nonan)

if 2 * len(phrase) < no_nan:
  print(no_nan)
    # features = {
    #     FEATURE_COLUMNS[i]: tf.train.Feature(
    #         float_list=tf.train.FloatList(value=frames[:, i])
    #     )
    #     for i in range(len(FEATURE_COLUMNS))
    # }
    # features["phrase"] = tf.train.Feature(
    #     bytes_list=tf.train.BytesList(value=[bytes(phrase, "utf-8")])
    # )
    # record_bytes = tf.train.Example(
    #     features=tf.train.Features(feature=features)
    # ).SerializeToString()

# COMMAND ----------

np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)

# COMMAND ----------

display(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1))

# COMMAND ----------

len(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1))

# COMMAND ----------

display(np.isnan(frames[:, RHAND_IDX]))

# COMMAND ----------

# Set length of frames to 128
FRAME_LEN = 128

# Create directory to store the new data
if not os.path.isdir("preprocessed"):
    os.mkdir("preprocessed")
else:
    shutil.rmtree("preprocessed")
    os.mkdir("preprocessed")

# Loop through each file_id
for file_id in tqdm(dataset_df.file_id.unique()):
    # Parquet file name
    pq_file = f"/kaggle/input/asl-fingerspelling/train_landmarks/{file_id}.parquet"
    # Filter train.csv and fetch entries only for the relevant file_id
    file_df = dataset_df.loc[dataset_df["file_id"] == file_id]
    # Fetch the parquet file
    parquet_df = pq.read_table(
        f"/kaggle/input/asl-fingerspelling/train_landmarks/{str(file_id)}.parquet",
        columns=["sequence_id"] + FEATURE_COLUMNS,
    ).to_pandas()
    # File name for the updated data
    tf_file = f"preprocessed/{file_id}.tfrecord"
    parquet_numpy = parquet_df.to_numpy()
    # Initialize the pointer to write the output of
    # each `for loop` below as a sequence into the file.
    with tf.io.TFRecordWriter(tf_file) as file_writer:
        # Loop through each sequence in file.
        for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
            # Fetch sequence data
            frames = parquet_numpy[parquet_df.index == seq_id]

            # Calculate the number of NaN values in each hand landmark
            r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
            l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
            no_nan = max(r_nonan, l_nonan)

            if 2 * len(phrase) < no_nan:
                features = {
                    FEATURE_COLUMNS[i]: tf.train.Feature(
                        float_list=tf.train.FloatList(value=frames[:, i])
                    )
                    for i in range(len(FEATURE_COLUMNS))
                }
                features["phrase"] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[bytes(phrase, "utf-8")])
                )
                record_bytes = tf.train.Example(
                    features=tf.train.Features(feature=features)
                ).SerializeToString()
                file_writer.write(record_bytes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get the saved TFRecord files into a list

# COMMAND ----------

tf_records = dataset_df.file_id.map(lambda x: f'/kaggle/working/preprocessed/{x}.tfrecord').unique()
print(f"List of {len(tf_records)} TFRecord files.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load character_to_prediction json file
# MAGIC
# MAGIC This json file contains a character and its value. We will add three new characters, "<" and ">" to mark the start and end of each phrase, and "P" for padding.

# COMMAND ----------

with open ("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
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
# MAGIC ### Create function to parse data from TFRecord format
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
# MAGIC ### Create function to convert the data 
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
# MAGIC ### Train and validation split/Create the final datasets
# MAGIC
# MAGIC Use the functions we defined above to create the final dataset.

# COMMAND ----------

batch_size = 64
train_len = int(0.8 * len(tf_records))

train_ds = tf.data.TFRecordDataset(tf_records[:train_len]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
valid_ds = tf.data.TFRecordDataset(tf_records[train_len:]).map(decode_fn).map(convert_fn).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


