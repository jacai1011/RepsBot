import numpy as np
import cv2
import serial

center_tolerance = 5; 
 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (140, 140))
    boxes, weights = hog.detectMultiScale(frame, winStride=(1,1), scale = 1.05)
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    centers = []
    for box in boxes:
        center_x = ((box[2]-box[0])/2)+box[0]
        x_pos_rel_center = (center_x-70)
        dist_to_center_x = abs(x_pos_rel_center)
        centers.append({'box': box, 'x_pos_rel_center': x_pos_rel_center, 'dist_to_center_x':dist_to_center_x})    
    if len(centers) > 0:
        sorted_boxes = sorted(centers, key=lambda i: i['dist_to_center_x'])
        center_box = sorted_boxes[0]['box']
        for box in range(len(sorted_boxes)):
            if box == 0:
                cv2.rectangle(frame, (sorted_boxes[box]['box'][0],sorted_boxes[box]['box'][1]), (sorted_boxes[box]['box'][2],sorted_boxes[box]['box'][3]), (0,255, 0), 2)
            else:
                cv2.rectangle(frame, (sorted_boxes[box]['box'][0],sorted_boxes[box]['box'][1]), (sorted_boxes[box]['box'][2],sorted_boxes[box]['box'][3]),(0,0,255),2)
        Center_box_pos_x = sorted_boxes[0]['x_pos_rel_center']  
        if -center_tolerance <= Center_box_pos_x <= center_tolerance:
            print("center")
        elif Center_box_pos_x >= center_tolerance:
            print("right")
        elif Center_box_pos_x <= -center_tolerance:
            print("left")
        print(str(Center_box_pos_x))
    else:
        print("nothing detected")
    frame = cv2.resize(frame,(720,720))
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)