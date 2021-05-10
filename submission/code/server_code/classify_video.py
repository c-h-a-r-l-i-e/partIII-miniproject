"""
This program will classify the RAVDESS dataset, using one of several modes -
lstm / mean / mode. It takes a command line argument to represent the resolution of
video that should be used, and automatically classifies at 5, 15, and 30 fps, saving the
results as a binary file.
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
from scipy.stats import mode

import tensorflow as tf

# Setup GPUs to allow other usage
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


# One of RN50, fer-RN50, RN50-imagenet, RN50-vgg, or whole-vgg-lstm
MODEL = "whole-vgg-RN50-lstm"

# One of ravdess, ravdess-faces
DATASET = "ravdess-faces"

CLASSIFY = 'mode'

if CLASSIFY == "lstm":
    MODEL = "whole-vgg-RN50-lstm"

res = int(sys.argv[1])

for fps in [30, 15, 5]:

    if MODEL == "RN50":
        lb = pickle.loads(open("ravdess_label_bin", "rb").read())
        model = load_model('../models/best-models/resnet50.h5')
        grayscale = False

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

    if MODEL == "RN50-lstm":
        model = load_model("../models/best-models/resnet50-lstm.h5")
        with open('ravdess_label_bin', 'rb') as file:
            lb = pickle.load(file)

    if MODEL == "vgg-RN50-lstm":
        model = load_model("../models/best-models/vgg-resnet50-lstm.h5")
        with open('ravdess_label_bin', 'rb') as file:
            lb = pickle.load(file)

    if MODEL == "whole-vgg-RN50-lstm":
        model = load_model("../models/best-models/whole-vgg-resnet50-lstm.h5")
        with open('ravdess_label_bin', 'rb') as file:
            lb = pickle.load(file)


    def mean_classify(preds_list):
        results = np.array(preds_list).mean(axis=0)
        i = np.argmax(results)
        return i

    def mode_classify(preds_list):
        preds = np.argmax(preds_list, axis=1)
        result = mode(preds)[0][0]
        return result

    def classify(preds_list):
        if CLASSIFY == "mean":
            return mean_classify(preds_list)
        elif CLASSIFY == "mode":
            return mode_classify(preds_list)
        else:
            raise ValueError("invalid classsification {}".format(CLASSIFY))

    def get_preds_list(filename):
        mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

        frames = []
        vs = cv2.VideoCapture(filename)

        # Loop over video frames
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break

            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else: 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if MODEL in ["RN50-vgg", "RN50-imagenet"]:
                frame = cv2.resize(frame, (197, 197)).astype("float32")
                frame -= 128.8006

            else:
                frame = cv2.resize(frame, (224, 224)).astype("float32")

            frames.append(frame)
            

        preds_arr = model.predict(np.array(frames), batch_size=128)
            
        vs.release()
        return preds_arr

    vidlist = glob.glob("../data/dataset/{}/*-{}-{}.mp4".format(DATASET, fps, res))
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

    if MODEL == "whole-vgg-RN50-lstm":
        x = []
        y = []
        for i, video in enumerate(vidlist):
            emotion = emotion_dict[int(re.search('(?<=\/0)\d', video).group())]
            if emotion in lb.classes_:
                x.append(video)
                y.append(emotion)

        # split into test/train/validate
        train_x , test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

        actual = []
        predictions = []

        for i, video in enumerate(test_x):
            print("classifying video {} / {}".format(i, len(test_x)))
            emotion = test_y[i]
            frames = utils.data.get_frame_array(video)

            pred = np.argmax(model.predict(np.array([frames]))[0])

            predictions.append(pred)
            actual.append(emotion)


    elif MODEL == "vgg-RN50-lstm" or model == "RN50-lstm":
        num_frames = 32
        skip = 16
        batch_size = 32

        x = []
        y = []
        for i, video in enumerate(vidlist):
            emotion = emotion_dict[int(re.search('(?<=\/0)\d', video).group())]
            if emotion in lb.classes_:
                x.append(video)
                y.append(emotion)

        # split into test/train/validate
        train_x , test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

        actual = []
        predictions = []

        for i, video in enumerate(test_x):
            print("classifying video {} / {}".format(i, len(test_x)))
            emotion = test_y[i]
            batches = []
            frames = utils.data.get_frame_array(video)

            if len(frames) <= num_frames:
                extension = np.full((num_frames - len(frames) + 1, 197, 197, 3), frames[-1])
                frames = np.append(frames, extension, axis=0)

            for frame_i in range(0, len(frames) - num_frames, skip):
                batches.append(frames[frame_i : frame_i + num_frames])

            preds = model.predict(np.array(batches), batch_size = batch_size)

            pred = classify(preds)

            predictions.append(pred)
            actual.append(emotion)


    else:
        x = []
        y = []
        for i, video in enumerate(vidlist):
            emotion = emotion_dict[int(re.search('(?<=\/0)\d', video).group())]
            if emotion in lb.classes_:
                x.append(video)
                y.append(emotion)

        # split into test/train/validate
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        test_x = test_x

        actual = []
        predictions = []

        for i, video in enumerate(test_x):
            print("Processing video {}/{}.".format(i, len(test_x)))
            emotion = test_y[i]
            actual.append(emotion)

            preds_list = get_preds_list(video)
            pred = classify(preds_list)

            predictions.append(pred)


    actual = lb.transform(actual).argmax(axis=1)


    results_dict = {
            "actual" : actual,
            "predictions" : predictions,
            "lb" : lb}

    with open("results/{}-{}-{}-{}-{}.pickle".format(DATASET, MODEL, fps, res, CLASSIFY), "wb") as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(classification_report(actual,
            predictions, labels = range(8), target_names=lb.classes_))

