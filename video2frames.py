import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Same DataGen function as in ipynb file
dataGen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# Load Model
model = load_model('/Users/aryyamakumarjana/Downloads/pythonProject/ASLmodel.h5')

# Define constraints and classes
imgSize = 200
rectSize = 400
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
inWidth = 368
inHeight = 368
thr = 0.2
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Now taking input from webcam. To be modified to video input instead of 0 once palm detection algo is done
cap = cv2.VideoCapture("/Users/aryyamakumarjana/Downloads/sample.MOV")

while True:
    # Store Returns and frames from video
    ret, frame = cap.read()
    # Palm Detection 
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            if idTo == 4:
                cv2.rectangle(frame, (points[idTo][0]-50, points[idTo][1]-50), (points[idTo][0]+25, points[idTo][1]+10), (0, 255, 0), 3)
            if idTo == 7:
                cv2.rectangle(frame, (points[idTo][0]-25, points[idTo][1]-50), (points[idTo][0]+50, points[idTo][1]+10), (0, 255, 0), 3)

    # Set up frame for hand for testing, to be replaced by Posenet palm detection
    #cv2.rectangle(frame, (0, 0), (rectSize, rectSize), (0, 255, 0), 3)
    handFrame = frame[0:rectSize, 0:rectSize]
    handFrameResized = cv2.resize(handFrame, (imgSize, imgSize))
    handFrameReshaped = (np.array(handFrameResized)).reshape((1, imgSize, imgSize, 3))

    # Data Preprocessing
    preprocessedHandFrame = dataGen.standardize(np.float64(handFrameReshaped))

    # Class Prediction
    predictClass = np.array(model.predict(preprocessedHandFrame))
    predicted = classes[predictClass.argmax()]
    probability = predictClass[0, predictClass.argmax()]

    # Display predicted class and probability in frame
    cv2.putText(frame, '{} - {:.2f}%'.format(predicted, probability * 100),
                (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close webcam
cap.release()
cv2.destroyAllWindows()
