"""
This program will train a video classifier using a LSTM
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, AveragePooling2D, TimeDistributed, LSTM
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from imutils.video import count_frames
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import utils.data
import utils.model
import gc
import pickle
import glob
import re

# Setup GPUs to allow other usage
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


import wandb
from wandb.keras import WandbCallback
wandb.login()


with open('ravdess_label_bin', 'rb') as file:
    lb = pickle.load(file)
dataset = "ravdess-faces"
cnn_model = "../models/best_models/resnet50-vgg.h5"

vidlist = glob.glob("../data/dataset/{}/*-30-720.mp4".format(dataset))

emotion_dict = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }

x = []
y = []

for i, video in enumerate(vidlist):
    emotion = emotion_dict[int(re.search('(?<=\/0)\d', video).group())]
    if emotion in lb.classes_:
        x.append(video)
        y.append(emotion)

y = lb.transform(y)

# split into test/train/validate
train_x , test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
train_x , val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)


def get_model(num_frames):
    if cnn_model == "../models/best_models/resnet50-vgg.h5":
        base_model = load_model("../models/best-models/resnet50-vgg.h5")

        for layer in base_model.layers:
            layer.trainable = False

        # prune the top two layers from the model
        base_model._layers.pop()
        base_model._layers.pop()
        
        base_model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

        base_model = TimeDistributed(base_model, input_shape = [num_frames, 197, 197, 3]) # enable the CNN to be called for each frame

        model = Sequential([
            Input(shape = [num_frames, 197, 197, 3], ragged=True),
            base_model,
            LSTM(128),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(lb.classes_), activation='softmax')
        ])


    else:
        print("Model {} not supported yet, please implement that".format(cnn_model))
        
    return model


def train():
    # default hyperparameters
    config_defaults = {
        'batch_size' : 1,
        'epochs': 30,
        'skip' : 32
    }

    num_frames = None # set num_frames to None to load the entire video into the LSTM!

    wandb.init(project='sentiment', entity='charlieisalright', config=config_defaults)
    config = wandb.config

    config.architecture_name = "{} followed by LSTM".format(cnn_model)
    config.dataset = dataset

    # Compile the model
    opt = SGD()
    model = get_model(num_frames)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    print("fitting model")

    # Now we can start training!
    history = model.fit(
        utils.data.VideoSequence(train_x, train_y, config.batch_size, num_frames, config.skip),
        validation_data = utils.data.VideoSequence(val_x, val_y, config.batch_size, num_frames, 64),
        epochs = config.epochs,
        callbacks = [WandbCallback()]
    )

    return model

model = train()
