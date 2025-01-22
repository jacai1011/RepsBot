

import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
from tensorflow.lite.python.interpreter import Interpreter

# https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the webcam"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        print("Camera initiated.")
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True)
args = parser.parse_args()

MODEL_NAME = args.modeldir

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME)
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
output_stride = 32

# Below coords calculation functions based off
# https://github.com/ecd1012/rpi_pose_estimation/blob/main/run_pose_estimation.py
# edited to work with current setup
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_and_argmax2d(output_details, threshold):
    v1 = interpreter.get_tensor(output_details[0]['index'])[0]
    reshaped = sigmoid(np.reshape(v1, [-1, v1.shape[2]]))
    reshaped = (reshaped > threshold) * reshaped
    coords = np.argmax(reshaped, axis=0)
    yCoords = np.expand_dims(coords // v1.shape[1], 1)
    xCoords = np.expand_dims(coords % v1.shape[1], 1)
    return np.concatenate([yCoords, xCoords], 1)

def draw_lines(keypoints, image, bad_pts):
    color = (0, 255, 0)
    thickness = 2
    # https://www.tensorflow.org/lite/models/pose_estimation/overview
    body_map = [[5, 6], [5, 7], [7, 9], [5, 11], [6, 8], [8, 10], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
    for map_pair in body_map:
        if map_pair[0] in bad_pts or map_pair[1] in bad_pts:
            continue
        start = (int(keypoints[map_pair[0]][1]), int(keypoints[map_pair[0]][0]))
        end = (int(keypoints[map_pair[1]][1]), int(keypoints[map_pair[1]][0]))
        image = cv2.line(image, start, end, color, thickness)
    return image

try:
    print("Program started...")

    videostream = VideoStream(resolution=(257, 257), framerate=30).start()
    time.sleep(1)

    while True:
        frame1 = videostream.read()
        frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (257, 257))

        input_data = (np.float32(np.expand_dims(frame_resized, axis=0)) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        coords = sigmoid_and_argmax2d(output_details, 0.5)
        drop_pts = list(np.unique(np.where(coords == 0)[0]))
        keypoint_positions = coords * output_stride
        for i, pos in enumerate(keypoint_positions):
            if i not in drop_pts:
                cv2.circle(frame_resized, (int(pos[1]), int(pos[0])), 2, (0, 255, 0), 2)

        frame_resized = draw_lines(keypoint_positions, frame_resized, drop_pts)

        cv2.imshow('Keypoints Detection', frame_resized)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    videostream.stop()
