#import the necessary packages
import numpy as np
import cv2
import serial

#sets how many pixels away from the center a person needs to be before the head stops
center_tolerance = 5; 
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # resizing for faster detection
    frame = cv2.resize(frame, (140, 140))
    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(1,1), scale = 1.05)
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    centers = []
    for box in boxes:
        #get the distance from the center of each box's center x cord to the center of the screen and ad them to a list
        center_x = ((box[2]-box[0])/2)+box[0]
        x_pos_rel_center = (center_x-70)
        dist_to_center_x = abs(x_pos_rel_center)
        centers.append({'box': box, 'x_pos_rel_center': x_pos_rel_center, 'dist_to_center_x':dist_to_center_x})    
    if len(centers) > 0:
           #sorts the list by distance_to_center
        sorted_boxes = sorted(centers, key=lambda i: i['dist_to_center_x'])
        #draws the box
        center_box = sorted_boxes[0]['box']
        for box in range(len(sorted_boxes)):
        # display the detected boxes in the colour picture
            if box == 0:
                cv2.rectangle(frame, (sorted_boxes[box]['box'][0],sorted_boxes[box]['box'][1]), (sorted_boxes[box]['box'][2],sorted_boxes[box]['box'][3]), (0,255, 0), 2)
            else:
                cv2.rectangle(frame, (sorted_boxes[box]['box'][0],sorted_boxes[box]['box'][1]), (sorted_boxes[box]['box'][2],sorted_boxes[box]['box'][3]),(0,0,255),2)
        #retrieves the distance from center from the list and determins if the head should turn left, right, or stay put and turn lights on
        Center_box_pos_x = sorted_boxes[0]['x_pos_rel_center']  
        if -center_tolerance <= Center_box_pos_x <= center_tolerance:
            #turn on eye light
            print("center")
        elif Center_box_pos_x >= center_tolerance:
            #turn head to the right
            print("right")
        elif Center_box_pos_x <= -center_tolerance:
            #turn head to the left
            print("left")
        print(str(Center_box_pos_x))
    else:
        #prints out that no person has been detected
        print("nothing detected")
    #resizes the video so its easier to see on the screen
    frame = cv2.resize(frame,(720,720))
    # Display the resulting frame
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)