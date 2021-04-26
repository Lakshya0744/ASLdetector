import os
from os.path import join
import pandas as pd
from pandas import DataFrame
import time
from sklearn.metrics import classification_report
from statistics import mode
from alphabet_mode_main import predict_labels_from_frames, predict_words_from_frames


# get List of unhidden files
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

os.system('python3 clean.py')

print("What do you want to predict ?  \n1. Letters \n2. Words")

option = input( "Please select one of the choices: " )

# Initialise the prediction array
arrPred = []

# Letters
if option == '1':

    # Get list of test videos
    path_to_video = 'Videos_letters/Demo_videos'
    list_of_videos = listdir_nohidden( path_to_video )

    # Folder to save hand frames
    path_to_frames = 'handframes/Demo_frames'
    if not os.path.exists( path_to_frames ):
        os.makedirs( path_to_frames )

    # Initialise predicted array
    predicted = []
    
    for video in list_of_videos:

        path_to_file = join( path_to_video, video )
        print( "Test video " + video + " loaded" )
        
        os.system('python3 ./handtracking/detect_single_threaded.py --source=%s --video=%s --frame_path=%s' % (path_to_file, video, path_to_frames) )
        
        path_to_test = join( path_to_frames, video )
        arrPred = predict_labels_from_frames( path_to_test )
        
        # Calculate Prediction and handle none cases
        try:
            valPred = mode( arrPred )
        except:
            valPred = ''
        
        actualLabel = video[0]
        print("\nActual Value: " + actualLabel + " Predicted Value: " + valPred )
        predicted.append( [valPred, actualLabel] )

    df = DataFrame ( predicted, columns=['predicted', 'actual'] )
    print( classification_report( df.predicted, df.actual ) )
    df.to_csv( join( path_to_video, 'result.csv' ) )

# words
if option == '2':

    os.system('python3 "./posenet_nodejs_setup-master/Python Scripts/Frames_Extractor.py" - DWITH_FFMPEG = ON')
    os.system('node ./posenet_nodejs_setup-master/scale_to_videos.js')
    os.system('python3 "./posenet_nodejs_setup-master/Python Scripts/convert_to_csv.py"')

    # Get list of test videos
    path_to_video = 'Videos_Words/Demo_videos'
    list_of_videos = listdir_nohidden( path_to_video )

    # Folder to save hand frames
    path_to_frames = 'handframes/Demo_frames'
    if not os.path.exists( path_to_frames ):
        os.makedirs( path_to_frames )

    # Initialise predicted array
    predicted = []

    for video in list_of_videos:

        path_to_file = join( path_to_video, video )
        print( "Test video " + video + " loaded" )

        os.system('python3 ./handtracking/detect_single_threaded.py --source=%s --video=%s --frame_path=%s' % (path_to_file, video, path_to_frames) )

        keyptPosenet = pd.read_csv( './keypoints_Posenet/' + video + '.csv' )

        coordRWx = keyptPosenet.rightWrist_x
        coordRWy = keyptPosenet.rightWrist_y
        coordLWx = keyptPosenet.leftWrist_x
        coordLWy = keyptPosenet.leftWrist_y

        merge = []
        lastframe = -1


        for i in range(len(coordRWx)-1):
            lThres = 0.4
            movRWx = coordRWx[i + 1] - coordRWx[i]
            movRWy = coordRWy[i + 1] - coordRWy[i]
            movLWx = coordLWx[i + 1] - coordLWx[i]
            movLWy = coordLWy[i + 1] - coordLWy[i]
            if ( movRWx > lThres) or (movRWy > lThres) or (movLWx > lThres) or (movLWy > lThres):
                lastframe = i

            path_to_test = join( path_to_frames, video )
            arrPred = predict_words_from_frames( path_to_test, lastframe )

            try:
                valPred = mode( arrPred )
            except:
                valPred = ''

            merge.append( valPred )

        predword = ''.join( merge ).upper()
        actualLabel = video.split( "." )[0]

        print("\nPlease wait......\n")
        print("\nPrediction results will be available soon...")
        for i in range(0,5):
            if i == 2:
                print("You are almost there")
            print(".")
            time.sleep(1)

        print("\nTrue Value: " + actualLabel + " Prediction: " + predword )

        time.sleep(1)
        predicted.append( [predword, actualLabel] )

    df = DataFrame ( predicted, columns=['predicted', 'actual'] )
    print( classification_report( df.predicted, df.actual ) )
    df.to_csv( join( path_to_video, 'result.csv' ) )
