import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
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

# One of ravdess, ravdess-faces, fer+
dataset = "fer+"

# One of RN18-FER+, RN18-MS, RN50-vgg,  or RN50
EXPERIMENT = "RN50"

if EXPERIMENT in ['RN18-FER+', 'RN18-MS']:
    channels_first = True
else:
    channels_first = False

dims = (197, 197)
    
trainX, valX, testX, trainY, valY, testY, lb = utils.data.load_img_dataset(dataset, channels_first, dims = dims)


# Randomly change the train set so results are more generalizable
if EXPERIMENT in ['RN18-FER+', 'RN18-MS']:
    data_format = 'channels_first'
    train_augmentation = ImageDataGenerator(
        fill_mode="nearest",
        data_format=data_format)
else:
    data_format = 'channels_last'
    train_augmentation = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        shear_range=10,
        horizontal_flip=True,
        fill_mode="reflect",
        data_format=data_format)


def get_model():
    return utils.model.get_model(EXPERIMENT, len(lb.classes_))


def full_train():

    wandb.init(project='sentiment', entity='charlieisalright')

    batch_size = 64

    # First train the top layers
    opt = Adam(epsilon = 1e-08, decay = 0.0)
    model = get_model()
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])


    train_generator = train_augmentation.flow(
        trainX,
        trainY,
        batch_size  = batch_size)


    model.fit_generator(train_generator, steps_per_epoch = len(trainX) // batch_size, 
            epochs = 5, validation_data = (valX, valY), callbacks = [WandbCallback()])


    # Next we fine-tune all of the model

    for layer in model.layers:
        layer.trainable = True

    opt = SGD(lr = 1e-4, momentum=0.9, decay = 0.0, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])


    def scheduler(epoch):
        updated_lr = K.get_value(model.optimizer.lr) * 0.5
        if (epoch % 3 == 0) and (epoch != 0):
            K.set_value(model.optimizer.lr, updated_lr)
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    reduce_lr_plateau = ReduceLROnPlateau(
	monitor 	= 'val_loss',
	factor		= 0.5,
	patience	= 3,
	mode 		= 'auto',
	min_lr		= 1e-8)


    early_stop = EarlyStopping(
	monitor 	= 'val_loss',
	patience 	= 10,
	mode 		= 'auto')

    model.fit_generator(train_generator, steps_per_epoch = len(trainX) // batch_size, 
            epochs = 100, validation_data = (valX, valY), 
            callbacks = [WandbCallback(), reduce_lr, reduce_lr_plateau, early_stop])

    return model


def train():
    # default hyperparameters
    config_defaults = {
        'batch_size' : 64,
        'learning_rate' : 0.0008475,
        'epochs': 30,
        'momentum' : 0.9,
        'decay': 1e-4
    }

    wandb.init(project='sentiment', entity='charlieisalright', config=config_defaults)
    config = wandb.config

    config.architecture_name = EXPERIMENT
    config.dataset = dataset

    # Compile the model, using stochastic gradient descent optimization.
    opt = SGD(lr=config.learning_rate, momentum=config.momentum, decay=config.decay / config.epochs)
    model = get_model()
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # Now we can start training!
    H = model.fit(
        x = trainX,
        y = trainY,
        batch_size=config.batch_size,
        steps_per_epoch = len(trainX) // config.batch_size,
        validation_data = (valX, valY),
        validation_steps = len(valX) // config.batch_size,
        epochs = config.epochs,
        callbacks = [WandbCallback()]
    )

    return model

model = full_train()

"""
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3
        },
    "parameters":{
        "epochs": {
            "distribution": "int_uniform",
            "min": 13,
            "max": 50
        },
        "batch_size": {
            "distribution": "int_uniform",
            "min": 4,
            "max": 64
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.00001,
            "max": 0.01
        }
    }
}

print("Initializing sweep agent")

sweep_id = 'w4teschu' # wandb.sweep(sweep_config, project='sentiment')
wandb.agent(sweep_id, project='sentiment', function=train)

"""
