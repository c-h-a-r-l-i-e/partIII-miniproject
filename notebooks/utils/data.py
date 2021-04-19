from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
import pickle
import csv

def load_img_dataset(dataset, channels_first=True):
    if dataset not in ['ravdess', 'ravdess-faces', 'fer+']:
        raise ValueError("No such dataset {}".format(dataset))


    if dataset in ['ravdess', 'ravdess-faces']:
        image_paths = list(paths.list_images("../data/img_dataset/{}/".format(dataset)))
        data = []
        labels = []

        print("Processing {} images".format(len(image_paths)))

        for image_path in image_paths:
            label = image_path.split(os.path.sep)[-2]
            
            # Load images, converting to RGB channels and scaling to 224 x 224
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            if channels_first:
                image = image.transpose(2,0,1)
            
            data.append(image)
            labels.append(label)
            
        print("Done fetching images")
            
        data = np.array(data)
        labels = np.array(labels)

        # One-hot encoding
        if os.path.exists('ravdess_label_bin'):
            lb = pickle.loads(open("ravdess_label_bin", "rb").read())
            labels = lb.transform(labels)
            
            
        else:
            lb = LabelBinarizer()
            labels = lb.fit_transform(labels)
            
            # serialize the label binarizer to disk
            f = open("ravdess_label_bin", "wb")
            f.write(pickle.dumps(lb))
            f.close()

        (trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                test_size=0.2, stratify=labels, random_state=42)

        (trainX, valX, trainY, valY) = train_test_split(trainX, trainY, 
                                test_size=0.25, stratify=labels, random_state=42)

        return trainX, valX, testX, trainY, valY, testY, lb


    if dataset == "fer+":
        trainX, trainY = get_data_fer('/home/charlie/Documents/courses/miniproject/data/img_dataset/FERPlus/data/FER2013Train', channels_first)

        # One-hot encoding
        if os.path.exists('fer_label_bin'):
            lb = pickle.loads(open("fer_label_bin", "rb").read())
            trainY = lb.transform(trainY)
            
        else:
            lb = LabelBinarizer()
            trainY = lb.fit_transform(trainY)
            print(lb.classes_)

            # serialize the label binarizer to disk
            f = open("fer_label_bin", "wb")
            f.write(pickle.dumps(lb))
            f.close()

        valX, valY = get_data_fer('/home/charlie/Documents/courses/miniproject/data/img_dataset/FERPlus/data/FER2013Valid', channels_first)
        valY = lb.transform(valY)

        testX, testY = get_data_fer('/home/charlie/Documents/courses/miniproject/data/img_dataset/FERPlus/data/FER2013Test', channels_first)
        testY = lb.transform(testY)

        return trainX, valX, testX, trainY, valY, testY, lb



def get_data_fer(folder, channels_first): 
    data = []
    labels = []

    emotion_header = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'non-face']


    with open(os.path.join(folder, "label.csv")) as csvfile:
        emotion_label = csv.reader(csvfile)
        for row in emotion_label:

            # Load labels, using majority voting
            emotion_raw = list(map(float, row[2:len(row)]))

            sumlist = sum(emotion_raw)
            idx = np.argmax(emotion_raw)
            emotion = emotion_header[idx]

            # Check there is a majority concensus, and that it is not unknown or non-face (or contempt)
            if emotion_raw[idx] >= 0.5 * sumlist and emotion not in ['unknown', 'non-face', 'contempt']:
                image_path = os.path.join(folder, row[0])
                image = cv2.imread(image_path)

                # TODO: consider whether we should resize??!!
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))

                if channels_first:
                    image = image.transpose(2,0,1)

                data.append(image)
                labels.append(emotion)

    return np.array(data), np.array(labels)

                


