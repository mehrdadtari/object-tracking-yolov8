import cv2
from tracker import *
from yolo_detector import yolo_detector
from ultralytics import YOLO

# Create tracker object
tracker = EuclideanDistTracker()

# Create detector model
model = YOLO("yolov8m.pt")

# Select video stream
cap = cv2.VideoCapture("car.mp4")

# If you want to use a region of interest (roi), use with_roi = True
with_roi = False

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    if not ret:
        break

    # Extract Region of interest
    # set the percentage of height and width for roi.
    if with_roi:
        roi = frame[int(0.4 * height): int(0.7 * height), int(0 * width): int(1 * width)]
    else:
        roi = frame

    # 1. Object Detection
    roi, detections = yolo_detector(roi, model)

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 26), cv2.FONT_HERSHEY_PLAIN, 2, (225, 0, 0), 2)
    
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()