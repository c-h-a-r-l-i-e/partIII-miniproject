from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as np
import cv2
import pickle
import csv
from imutils.video import count_frames

def load_img_dataset(dataset, channels_first=True, four_emotions=False, dims = (224, 224)):
    print("input_dims [ {}".format(dims))

    four_emotion_list = ['angry', 'happy', 'sad', 'neutral']
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
            image = cv2.resize(image, dims)
            
            if channels_first:
                image = image.transpose(2,0,1)

            if four_emotions:
                if label in four_emotion_list:
                    data.append(image)
                    labels.append(label)

            else:
                data.append(image)
                labels.append(emotion)
            
        print("Done fetching images")
            
        data = np.array(data)
        labels = np.array(labels)

        if four_emotions:
            if os.path.exists('four_em_label_bin'):
                lb = pickle.loads(open('four_em_label_bin', "rb").read())
                labels = lb.transform(labels)
                
                
            else:
                lb = LabelBinarizer()
                labels = lb.fit_transform(labels)
                
                # serialize the label binarizer to disk
                f = open("four_em_label_bin", "wb")
                f.write(pickle.dumps(lb))
                f.close()

        else:
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

        (trainX, valX, trainY, valY) = train_test_split(data, labels, 
                                test_size=0.25, stratify=labels, random_state=42)

        testX, testY = [], []

        return trainX, valX, testX, trainY, valY, testY, lb


    if dataset == "fer+":
        trainX, trainY = get_data_fer('../data/img_dataset/FERPlus/data/FER2013Train', channels_first, four_emotions, dims)
        print("train dims : {}".format(dims))

        if four_emotions:
            if os.path.exists('four_em_label_bin'):
                lb = pickle.loads(open('four_em_label_bin', "rb").read())
                trainY = lb.transform(trainY)
                
                
            else:
                lb = LabelBinarizer()
                trainY = lb.fit_transform(trainY)
                
                # serialize the label binarizer to disk
                f = open("four_em_label_bin", "wb")
                f.write(pickle.dumps(lb))
                f.close()

        else:
            if os.path.exists('fer_label_bin'):
                lb = pickle.loads(open("fer_label_bin", "rb").read())
                lb.classes_ = ["angry", "disgust", "fearful", "happy", "sad", "surprised", "neutral"]
                
                trainY = lb.transform(trainY)
                
            else:
                lb = LabelBinarizer()
                trainY = lb.fit_transform(trainY)
                print(lb.classes_)

                # serialize the label binarizer to disk
                f = open("fer_label_bin", "wb")
                f.write(pickle.dumps(lb))
                f.close()

        print("val dims : {}".format(dims))
        valX, valY = get_data_fer('../data/img_dataset/FERPlus/data/FER2013Valid', channels_first, four_emotions, dims)
        valY = lb.transform(valY)

        testX, testY = get_data_fer('../data/img_dataset/FERPlus/data/FER2013Test', channels_first, four_emotions, dims)
        testY = lb.transform(testY)

        return trainX, valX, testX, trainY, valY, testY, lb



def get_data_fer(folder, channels_first, four_emotions, dims): 
    four_emotion_list = ['angry', 'happy', 'sad', 'neutral']
    data = []
    labels = []

    emotion_header = ['neutral', 'happy', 'surprised', 'sad', 'angry', 'disgust', 'fearful', 'contempt', 'unknown', 'non-face']


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
                image = cv2.resize(image, dims).astype("float32")
                image -= 128.8006

                if channels_first:
                    image = image.transpose(2,0,1)

                if four_emotions:
                    if emotion in four_emotion_list:
                        data.append(image)
                        labels.append(emotion)

                else:
                    data.append(image)
                    labels.append(emotion)

    return np.array(data), np.array(labels)

def get_frame_array(filename):
    frames = []

    vs = cv2.VideoCapture(filename)

    # Loop over video frames
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # convert to greyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        frame = cv2.resize(frame, (197, 197)).astype("float32")
        frame -= 128.8006
        # frame /= 64.6497
        frames.append(frame)

    return np.array(frames)

                
class VideoSequence(Sequence):
    """
    This video sequence is a generator that returns a selection of frames from a video.
    It only loads videos into memory one at a time, avoiding excessive memory usage.
    """
    def __init__(self, xs, ys, batch_size, num_frames, skip, num_samples = None):
        self.xs, self.ys = xs, ys
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.vid_num = 0
        self.vid_index = 0
        self.skip = skip

        if num_frames is None:
            self.num_samples = len(self.xs)
        else:
            if num_samples is None:
                self.num_samples = self._num_samples()
            else:
                self.num_samples = num_samples

    def get_frame_array(self, filename):
        frames = []

        vs = cv2.VideoCapture(filename)

        # Loop over video frames
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break

            # convert to greyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            frame = cv2.resize(frame, (197, 197)).astype("float32")
            frame -= 128.8006
            #frame /= 64.6497
            frames.append(frame)

        return np.array(frames)


    def _num_samples(self):
        num_samples = 0
        for filename in self.xs:
            frame_cnt = count_frames(filename)
            num_samples += (frame_cnt - self.num_frames + 1) // self.skip
        print("counted num samples = {}".format(num_samples))
        return num_samples

    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, idx):
        frames_out = []
        classes_out = []
        if self.num_frames is None:
            if idx == 0:
                self.vid_num = 0

            for batch_num in range(self.batch_size):
                if self.vid_num >= len(self.xs):
                    self.vid_num = 0
                    print("WARNING: having to reuse data")
                    self.vid_index = 30
                frames = self.get_frame_array(self.xs[self.vid_num]).astype(np.float32)
                y = self.ys[self.vid_num]

                frames_out.append(frames)
                classes_out.append(y)

                self.vid_num += 1

        else:
            if idx == 0:
                self.vid_num = 0
                self.vid_index = 0 
                self.frames = self.get_frame_array(self.xs[self.vid_num])
                self.y = self.ys[self.vid_num]

            for batch_num in range(self.batch_size):
                if self.vid_index + self.num_frames > len(self.frames):
                    self.vid_num += 1
                    self.vid_index = 0
                    if self.vid_num >= len(self.xs):
                        self.vid_num = 0
                        print("WARNING: having to reuse data")
                        self.vid_index = 30
                    self.frames = self.get_frame_array(self.xs[self.vid_num]).astype(np.float32)
                    self.y = self.ys[self.vid_num]

                frames_out.append(self.frames[self.vid_index: self.vid_index + self.num_frames])
                classes_out.append(self.y)
                self.vid_index += self.skip


        frames_out = np.array(frames_out)        
        classes_out = np.array(classes_out)

        return frames_out, classes_out 

