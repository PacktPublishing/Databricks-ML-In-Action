# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 4: Exploring and Cleaning Toward a Silver Layer
# MAGIC
# MAGIC ## ASL Fingerspelling - ASL Exploring
# MAGIC
# MAGIC [Kaggle competition link](https://www.kaggle.com/competitions/asl-fingerspelling/)
# MAGIC

# COMMAND ----------

# MAGIC %md ##Run Setup

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=asl-fingerspelling $catalog=lakehouse_in_action

# COMMAND ----------

# MAGIC %md Viewing the data with SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM train_landmarks LIMIT 100

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM supplemental_landmarks LIMIT 100

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data visualization
# MAGIC
# MAGIC The starter notebook provides useful functions. We will use those functions to visualize the landmarks. The landmarks are coordinates of specific points on the human hand.
# MAGIC
# MAGIC Reference: [Data visualization using mediapipe APIs by sknadig](https://www.kaggle.com/code/nadigshreekanth/data-visualization-using-mediapipe-apis)

# COMMAND ----------

from matplotlib import animation, rc
from mediapipe.framework.formats import landmark_pb2

import matplotlib
import matplotlib.pyplot as plt
import mediapipe
import numpy as np
import pandas as pd


# COMMAND ----------

# MAGIC %md
# MAGIC ### Competition provided functions
# MAGIC

# COMMAND ----------

# Animation from images.

matplotlib.rcParams['animation.embed_limit'] = 2**128
matplotlib.rcParams['savefig.pad_inches'] = 0
matplotlib.rc('animation', html='jshtml')

def create_animation(images):
    fig = plt.figure(figsize=(6, 9))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im=ax.imshow(images[0], cmap="gray")
    plt.close(fig)
    
    def animate_func(i):
        im.set_array(images[i])
        return [im]

    return matplotlib.animation.FuncAnimation(fig, animate_func, frames=len(images), interval=1000/10)

# COMMAND ----------

# Extract the landmark data and convert it to an image using medipipe library.
# This function extracts the data for both hands.

mp_pose = mediapipe.solutions.pose
mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils 
mp_drawing_styles = mediapipe.solutions.drawing_styles

def get_hands(seq_df):
    images = []
    all_hand_landmarks = []
    for seq_idx in range(len(seq_df)):
        x_hand = seq_df.iloc[seq_idx].filter(regex="x_right_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_right_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_right_hand.*").values

        right_hand_image = np.zeros((600, 600, 3))

        right_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        for x, y, z in zip(x_hand, y_hand, z_hand):
            right_hand_landmarks.landmark.add(x=x, y=y, z=z)

        mp_drawing.draw_landmarks(
                right_hand_image,
                right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        x_hand = seq_df.iloc[seq_idx].filter(regex="x_left_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_left_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_left_hand.*").values
        
        left_hand_image = np.zeros((600, 600, 3))
        
        left_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_hand, y_hand, z_hand):
            left_hand_landmarks.landmark.add(x=x, y=y, z=z)

        mp_drawing.draw_landmarks(
                left_hand_image,
                left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        images.append([right_hand_image.astype(np.uint8), left_hand_image.astype(np.uint8)])
        all_hand_landmarks.append([right_hand_landmarks, left_hand_landmarks])
    return images, all_hand_landmarks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizing the first sequence in the training table

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT sequence_id FROM train_landmarks LIMIT 1

# COMMAND ----------

example_df = sql("SELECT * FROM train_landmarks WHERE sequence_id='1983865658'")
example_df = example_df.select("sequence_id", *FEATURE_COLUMNS).toPandas()

# COMMAND ----------

# MAGIC %md Now we are using the Mediapipe library to view the animation of the hands

# COMMAND ----------

# Get the images created using mediapipe apis
hand_images, hand_landmarks = get_hands(example_df)
# Fetch and show the data for right hand
create_animation(np.array(hand_images)[:, 0])
