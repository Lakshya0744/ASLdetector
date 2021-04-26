import glob
import sys

import cv2
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor
import torch


def get_inference_vector_one_frame_alphabet(files_list):
    # model trained based on https://www.kaggle.com/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

    model = HandShapeFeatureExtractor.get_instance()
    vectors = []
    video_names = []
    step = int(len(files_list) / 100)
    if step == 0:
        step = 1

    count = 0
    for video_frame in files_list:
        # print(video_frames)
        # assert len(video_frames) == 6
        img = cv2.imread(video_frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = model.extract_feature(img)
        results = np.squeeze(results)
        predicted = np.where(results==max(results))[0][0]

        vectors.append(predicted)
        video_names.append(os.path.basename(video_frame))

        count += 1
        if count % step == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    return vectors

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def load_label_dicts(label_file):
    id_to_labels = load_labels(label_file)
    labels_to_id = {}
    i = 0

    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id


def predict_labels_from_frames(video_folder_path):
    
    files = []
    # wildcard to select all frames for given video file
    
    path = os.path.join(video_folder_path, "*.png")
    frames = glob.glob(path)
    
    # sort image frames
    frames.sort()
    files = frames
    
    prediction_vector = get_inference_vector_one_frame_alphabet(files)
    
    label_file = 'labels.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)
    
    final_predictions=[]
    
    for i in range(len(prediction_vector)):
        for ins in labels_to_id:
            if prediction_vector[i] == labels_to_id[ins]:
                final_predictions.append(ins)
    
    return final_predictions

def predict_words_from_frames(video_folder_path, till):
    
    files = []
    # wildcard to select all frames for given video file
    
    path = os.path.join(video_folder_path, "*.png")
    frames = glob.glob(path)
    
    # sort image frames
    frames.sort()
    files = frames[:till]
    
    prediction_vector = get_inference_vector_one_frame_alphabet(files)
    
    label_file = 'labels.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)
    
    final_predictions=[]
    
    for i in range(len(prediction_vector)):
        for ins in labels_to_id:
            if prediction_vector[i] == labels_to_id[ins]:
                final_predictions.append(ins)
    
    return final_predictions