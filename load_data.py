import os
import cv2
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from feature_extract import lbp_histogram
import numpy as np

RANDOM_SEED = 21  # Random seed for consistent data splits
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 1e-4
N_EPOCHS = 300
N_FRAMES = 10
PATH = "train/frames"


def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  frame_dir = os.path.join("train/frames",video_path.split("/")[2].split(".")[0]) 
  os.mkdir(frame_dir)
  cv2.imwrite(os.path.join(frame_dir , "0.jpg"), frame)
  print(frame_dir + "0.jpg")
  for idx in range(1, n_frames):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      cv2.imwrite(os.path.join(frame_dir , str(idx) + ".jpg"), frame)
  src.release()

def read_annotations(annotations_path):
  df = pd.read_csv(annotations_path)
  annotations = []
  idx = 0
  for i in df:
    if idx == 0:      
      for j in range(len(df)):
        annotation = []
        annotation.append(df[i][j])
        annotations.append(annotation)
    else:
      for j in range(len(df)):
        annotations[j].append(df[i][j])
    idx += 1
  return annotations

def get_frames_annotations(annotations):
  for annotation_idx in range(len(annotations)):
    name_frame = annotations[annotation_idx][0].split(".")[0]
    annotations[annotation_idx][0] = name_frame
  return annotations

annotations = read_annotations("train/label.csv")

def split(annotations, random_state, valid_size):
  train_annotations, valid_annotations = train_test_split(
      annotations, test_size = valid_size, random_state = random_state
  )
  return train_annotations, valid_annotations

train_annotations, valid_annotations = split(
    annotations, random_state= RANDOM_SEED, valid_size= 0.1
)

from pathlib import Path
class HistogramGenerator:
  def __init__(self, path, annotations, image_size):
  
    self.path = path
    self.image_size = image_size
    self.annotations = annotations

  def __call__(self):
    for annotation in self.annotations:
      frames_path = os.path.join(self.path, annotation[0])
      n_frames = len(os.listdir(frames_path))
      label = annotation[1]
      for frame_name in os.listdir(frames_path):
        image_path = os.path.join(frames_path, frame_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_h = lbp_histogram(image[:,:,0]) # y channel
        cb_h = lbp_histogram(image[:,:,1]) # cb channel
        cr_h = lbp_histogram(image[:,:,2]) # cr channel
        feature = np.concatenate((y_h,cb_h,cr_h))
        feature = tf.convert_to_tensor(feature, dtype=tf.float32)
        label = tf.convert_to_tensor(int(label), dtype=tf.float32)
        yield feature, label

# Create the training set
output_signature = (tf.TensorSpec(shape = (177), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
his_train_ds = tf.data.Dataset.from_generator(HistogramGenerator(PATH, train_annotations, IMAGE_SIZE),
                                          output_signature = output_signature)
his_val_ds = tf.data.Dataset.from_generator(HistogramGenerator(PATH, valid_annotations, IMAGE_SIZE),
                                          output_signature = output_signature)

# Print the shapes of the data
train_frames, train_labels = next(iter(his_train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(his_val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

his_train_ds = his_train_ds.batch(32)
his_val_ds = his_val_ds.batch(32)