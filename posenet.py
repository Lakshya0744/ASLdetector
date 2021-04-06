import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from pose_engine import PoseEngine
from PIL import Image
from PIL import ImageDraw

import numpy as np
import os




# Same DataGen function as in ipynb file
dataGen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

# Load Model
model = load_model('/Users/lgarg/Downloads/ASLmodel.h5')
# model = load_model('/Users/aryyamakumarjana/Downloads/pythonProject/ASLmodel.h5')

# Define constraints and classes
imgSize = 200
rectSize = 400
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']



# Now taking input from webcam. To be modified to video input instead of 0 once palm detection algo is done
cap = cv2.VideoCapture(0)

while True:
#     # Store Returns and frames from video
    ret, frame = cap.read()
    # print(frame)
    # Palm Detection - TBD
    im = Image.fromarray(frame)
    pil_image = im.convert('RGB')
    print(pil_image)
    engine = PoseEngine(
        '/Users/lgarg/Desktop/posenet/models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')
    poses, inference_time = engine.DetectPosesInImage(pil_image)
    print(poses)


# posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite





    # Set up frame for hand for testing, to be replaced by Posenet palm detection
    cv2.rectangle(frame, (0, 0), (rectSize, rectSize), (0, 255, 0), 3)
    print(frame)
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
