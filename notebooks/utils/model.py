from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


def get_RN50_model(num_labels):
    base_model = ResNet50(weights="imagenet", include_top=False, 
                      input_tensor=Input(shape=(224, 224, 3)))

    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(num_labels, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    # Ensure we don't train the base model
    for layer in base_model.layers:
        layer.trainable = False
        
    return model

def get_fer_model(num_labels):
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
    head_model = Dense(num_labels, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    
    # Ensure we don't train the base model
    for layer in base_model.layers:
        layer.trainable = False
    return model


def get_ms1m_model(num_labels):
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
    head_model = Dense(num_labels, activation="softmax")(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)
    
    # Ensure we don't train the base model
    for layer in base_model.layers:
        layer.trainable = False
    return model

def get_model(experiment, num_labels):
    if experiment == "RN18-FER+":
        return get_fer_model(num_labels)

    elif experiment == "RN18-MS":
        return get_ms1m_model(num_labels)
    
    elif experiment == "RN50":
        return get_RN50_model(num_labels)
    
    else:
        raise ValueError("Invalid EXPERIMENT setting : {}".format(EXPERIMENT))

