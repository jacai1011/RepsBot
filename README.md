# RepsCounter
Software for Raspberry Pi 4B used to automate track gym reps and posture for detected exercise using OpenCV. 

OpenCV required packages:

sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev  

sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev  

sudo apt-get -y install libxvidcore-dev libx264-dev  

sudo apt-get -y install qt4-dev-tools libatlas-base-dev


## HOG Tracker
Uses HOG (Histogram of oriented gradients) to track reps based on whether person is in detected frame or not. Only works on upright/standing exercises and counts based on detection of person based on whether they are standing or performing movement. Unreliable due to limitations of RPI4B and limited accuracy along with the limitations of the HOG itself being unable to track precise movements with given input.

## TFlow Tracker
Uses TensorFlow and MobileNet model to track precise movement of person. Able to map basic keypoints of a human frame output into a live stream. Limited to 257x257 due to RPI4B capability however tracks movement with reasonable reliability and accuracy. Uses the Google Posenet model:

wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite

Math for keypoint calculations based on:

https://ecd1012.medium.com/pose-estimation-on-the-raspberry-pi-4-83a02164eb8e

