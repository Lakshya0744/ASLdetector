import json
import time

import numpy as np
import pandas as pd
import os

path_to_videos = "/Users/aryyamakumarjana/Downloads/ASLFingerspellingPrediction/Videos_Words/Demo_videos"
path_to_frames = "/Users/aryyamakumarjana/Downloads/ASLFingerspellingPrediction/posenet_nodejs_setup-master/Posenet_Frames/"
path_to_keypoints = "/Users/aryyamakumarjana/Downloads/ASLFingerspellingPrediction/keypoints_Posenet/"

# get List of unhidden files
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

# Convert keypoints.json to csv
def convert_to_csv(path_to_frame, name):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    file_path = path_to_frame + name + '.mov.json'
    while not os.path.exists(file_path):
        time.sleep(1)
    data = json.loads(open(file_path, 'r').read())


    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    pd.DataFrame(csv_data, columns=columns).to_csv(path_to_keypoints + name + '.mov.csv', index_label='Frames#')


if __name__ == '__main__':
    # new_path = path_to_videos+'A/'
    files = listdir_nohidden(path_to_videos)
    for file in files:
        if not os.path.isdir(path_to_videos + "/" + file + "/"):
            name = os.path.splitext( file )[0]
            # new_path = path_to_videos + "/" + name + "/"
            new_path = path_to_frames + "/" + name + "/"
            convert_to_csv(new_path, name)
