# Import necessary libraries
from ultralytics import YOLO
import cv2
import cvzone
from math import ceil
import numpy as np

# Import the SORT (Simple Online and Realtime Tracking) algorithm script for object tracking
# https://github.com/abewley/sort/blob/master/sort.py
from sort import *

# Open the video file for processing
cap = cv2.VideoCapture("directory/images/video.mp4") #Change the path to your testing video

# Initialize YOLO model with trained weight
model = YOLO("D:/directory/Weights/best_100.pt")

# Define class names for detected objects
classNames = ['car', 'pickup', 'camping car', 'truck', 'others', 'tractor', 'boat', 'vans', 'motorcycles', 'buses', 'Small Land Vehicles', 'Large Land Vehicles']

# Initialize the SORT tracker with specified parameters
tracker = Sort(max_age=20, min_hits=20, iou_threshold=0.3)

# Open a text file for writing output data
output_file = open("output.txt", "w+")

Confidence=[]

while True:
    # Read a frame from the video
    success, img = cap.read()

    # Perform object detection using YOLO on the current frame
    results = model(img)

    # Write the detection results to the output file
    output_file.write(f'{results}\n')

    # Create an empty array to store bounding box coordinates
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Extract class and confidence score from the YOLO results
            cls = int(box.cls[0])
            conf = (ceil(box.conf[0] * 100)) / 100

            Confidence.append(conf)

            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width (b) and height (h) of the bounding box
            b, h = x2 - x1, y2 - y1

            # Calculate center coordinates (cx, cy) of the bounding box
            cx, cy = ((x1 + x2) // 2), ((y1 + y2) // 2)

            # Draw bounding box and label on the image
            cvzone.cornerRect(img, (x1, y1, b, h), l=5, t=2, rt=2)
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1 + 5, y1 - 7), offset=4, scale=0.8, thickness=2, font= cv2.FONT_HERSHEY_SIMPLEX)
            cv2.circle(img, (cx, cy), 4, (255, 255, 0), thickness=2)

            # Create an array to store the current object's information (x1, y1, x2, y2, confidence)
            currentArray = np.array([x1, y1, x2, y2, conf])

            # Append the current object's information to the detections array
            detections = np.vstack((detections, currentArray))

    # Update object tracking using the SORT algorithm
    resultsTracker = tracker.update(detections)

    # Resize the image for display
    h, w = img.shape[0:2]
    neww = 1200
    newh = int(neww * (h / w))
    img = cv2.resize(img, (neww, newh))

    # Display the image with object detection and tracking results
    cv2.imshow("Image", img)

    # Press 'q' to exit the frame window
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Close the output file, release the video capture, and close any open windows
output_file.close()
cap.release()
cv2.destroyAllWindows()

Model_Conf = sum(Confidence)/len(Confidence)
print("\n",Model_Conf)