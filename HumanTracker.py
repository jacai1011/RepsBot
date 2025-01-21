import numpy as np
import cv2
import time

# Initialize the HOG descriptor with the default people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam
cap = cv2.VideoCapture(0)

THRESHOLD_HEIGHT = 170
SUCCESS_DURATION = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the webcam.")
        break

    frame = cv2.resize(frame, (200, 200))

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.05)
    
    if len(boxes) > 0:
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        centers = [((box[2] - box[0]) / 2 + box[0] - 70, box) for box in boxes]
        sorted_boxes = sorted(centers, key=lambda c: abs(c[0]))
        center_box = sorted_boxes[0][1]
        
        height = center_box[3] - center_box[1]
        if height > THRESHOLD_HEIGHT:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= SUCCESS_DURATION and not success_printed:
                print("success") 
                success_printed = True

            cv2.rectangle(frame, (center_box[0], center_box[1]), (center_box[2], center_box[3]), (0, 255, 0), 2)

            print("Detected: Height =", height)
        else:
            start_time = None
            print("Detected: Height =", height)
    

        
            
    else:
        start_time = None
        success_printed = False
        print("Nothing detected")

    # Resize and display the frame only if a person is detected
    frame = cv2.resize(frame, (720, 720))
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
