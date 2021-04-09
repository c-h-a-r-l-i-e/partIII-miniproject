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
from tensorflow.keras.optimizers import SGD
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

import wandb
from wandb.keras import WandbCallback
wandb.login()

# Setup the location of the various parameters needed

# One of RN18-FER+, RN18-MS, or RN50
EXPERIMENT = "RN18-FER+"

dataset = "../data/img_dataset"
epochs = 25

labels = set(["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])

image_paths = list(paths.list_images(dataset))
random.seed(10)
random.shuffle(image_paths)
image_paths = image_paths[:3000]
data = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    
    # Load images, converting to RGB channels and scaling to 224 x 224
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    
    if EXPERIMENT in ['RN18-FER+', 'RN18-MS']:
        image = image.transpose(2,0,1)
    
    data.append(image)
    labels.append(label)
    
print("Done fetching images")
    
    
data = np.array(data)
labels = np.array(labels)

# One-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, 
                        test_size=0.25, stratify=labels, random_state=42)

# Randomly change the train set so results are more generalizable
        
if EXPERIMENT in ['RN18-FER+', 'RN18-MS']:
    data_format = 'channels_first'
else:
    data_format = 'channels_last'
train_augmentation = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    data_format=data_format)

val_augmentation = ImageDataGenerator(data_format=data_format)
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
train_augmentation.mean = mean
val_augmentation.mean = mean

print("Finished augmenting images")


def get_RN50_model():
    base_model = ResNet50(weights="imagenet", include_top=False, 
                      input_tensor=Input(shape=(224, 224, 3)))

    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(len(lb.classes_), activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    # Ensure we don't train the base model
    for layer in base_model.layers:
        layer.trainable = False
        
    return model


def get_fer_model():
    base_model = load_model('../models/fan-fer', compile=False)
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    
    head_model = base_model.layers[-1].output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(len(lb.classes_), activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    
    # Ensure we don't train the base model
    for layer in base_model.layers:
        layer.trainable = False
    return model


def get_ms1m_model():
    base_model = load_model('../models/fan-ms1m', compile=False)
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    base_model._layers.pop()
    
    head_model = base_model.layers[-1].output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(len(lb.classes_), activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    
    # Ensure we don't train the base model
    for layer in base_model.layers:
        layer.trainable = False
    return model

def get_model():
    if EXPERIMENT == "RN18-FER+":
        return get_fer_model()

    elif EXPERIMENT == "RN18-MS":
        return get_ms1m_model()
    
    elif EXPERIMENT == "RN50":
        return get_RN50_model()
    
    else:
        raise ValueError("Invalid EXPERIMENT setting : {}".format(EXPERIMENT))


def train():
    # default hyperparameters
    config_defaults = {
        'batch_size' : 64,
        'learning_rate' : 0.001,
        'epochs': 25
    }

    wandb.init(project='sentiment', entity='charlieisalright', config=config_defaults)
    config = wandb.config

    config.architecture_name = "ResNet18"
    config.dataset = "FER+"

    # Compile the model, using stochastic gradient descent optimization.
    opt = SGD(lr=config.learning_rate, momentum=0.9, decay=1e-4 / config.epochs)
    model = get_model()
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # Now we can start training!
    H = model.fit(
        x = train_augmentation.flow(trainX, trainY, batch_size=config.batch_size),
        steps_per_epoch = len(trainX) // config.batch_size,
        validation_data = val_augmentation.flow(testX, testY),
        validation_steps = len(testX) // config.batch_size,
        epochs = config.epochs,
        callbacks = [WandbCallback()]
    )


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

train()

print("Initializing sweep agent")

sweep_id = 'bdsacixu' # wandb.sweep(sweep_config, project='sentiment')
wandb.agent(sweep_id, project='sentiment', function=train)

