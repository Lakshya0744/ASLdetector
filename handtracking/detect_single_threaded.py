from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
from random import randint
import ffmpeg

detection_graph, sess = detector_utils.load_inference_graph()

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    print(path_video_file)
    meta_dict = ffmpeg.probe(path_video_file)

    rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
    return int(rotate)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.35,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-frame',
        '--frame_path',
        dest='frame_path',
        default=0,
        help='give a frame folder path')
    parser.add_argument(
        '-filename',
        '--video',
        dest='video_name',
        default=0,
        help='video name')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    rotateCode = check_rotation(args.video_source)
    #print(rotateCode)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # if rotateCode==90 or rotateCode==270:
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.height)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.width)
    # else:

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    
    ret= True
    count=0
    while ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        temp_image= image_np
        if ret:
            # image_np = cv2.flip(image_np, 1)
            if not image_np is None:
                try:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                except:
                    print("Error converting to RGB")

            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(image_np,
                                                          detection_graph, sess)
            
            count += 1
            # draw bounding boxes on frame
            detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                             scores, boxes, im_width, im_height,
                                             image_np, temp_image, count, args.frame_path, args.video_name, rotateCode)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            if (args.display > 0):
                # Display FPS on frame
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

                cv2.imshow('Single-Threaded Detection',
                           cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                print("frames processed: ", num_frames, "elapsed time: ",
                      elapsed_time, "fps: ", str(int(fps)))
        else:
            continue
