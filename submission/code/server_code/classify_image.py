"""
This program will test the classification performance of a model on the FER+ test set,
and save the results as a pickled file.
"""

from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import argparse
import pickle
import cv2
import glob
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sys
import utils.data
from tensorflow.keras.utils import Sequence

import tensorflow as tf


# Setup GPUs to allow other usage
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


# One of RN50, fer-RN50, RN50-imagenet or RN50-vgg
MODEL = "RN50-imagenet"

if MODEL == "RN50":
    lb = pickle.loads(open("ravdess_label_bin", "rb").read())
    model = load_model('../models/best-models/resnet50.h5')
    greyscale = False

if MODEL == "fer-RN50":
    lb = pickle.loads(open("fer_label_bin", "rb").read())
    model = load_model('../models/best-models/fer-resnet50.h5')
    grayscale = True

if MODEL == "RN50-vgg":
    model = load_model("../models/best-models/resnet50-vgg.h5")
    lb = pickle.loads(open("fer_label_bin", "rb").read())
    lb.classes_ = ["angry", "disgust", "fearful", "happy", "sad", "surprised", "neutral"]
    grayscale = True

if MODEL == "RN50-imagenet":
    model = load_model("../models/best-models/resnet50-imagenet.h5")
    lb = pickle.loads(open("fer_label_bin", "rb").read())
    lb.classes_ = ["angry", "disgust", "fearful", "happy", "sad", "surprised", "neutral"]
    grayscale = True


dims = (197, 197)
    
print("loading data")
testX, testY = utils.data.get_data_fer('../data/img_dataset/FERPlus/data/FER2013Test', False, False, dims=dims)
testY = lb.transform(testY)
print("data loaded")

predictions = model.predict(x=testX.astype("float32"), batch_size=128)
predictions = predictions.argmax(axis=1)

actual = testY.argmax(axis=1)

results_dict = {
        "actual" : actual,
        "predictions" : predictions,
        "lb" : lb}

with open("results/my-imagenet-rn50.pickle", "wb") as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(classification_report(actual,
        predictions, labels = range(8), target_names=lb.classes_))

