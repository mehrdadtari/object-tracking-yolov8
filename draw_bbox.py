import cv2
import numpy as np

# Thi function draws bounding boxes with the class names.
def bbox_drawer(res, frame, model):
    detections = []
    bboxes = np.array(res.boxes.xyxy.cpu(), dtype="int")
    confidence = np.array(res.boxes.conf.cpu())
    classes = np.array(res.boxes.cls.cpu(), dtype="int")

    for cls, bbox, conf in zip(classes, bboxes, confidence):
        (x, y, x2, y2) = bbox
        width = x2 - x
        height = y2 - y
        detections.append([x, y, width, height])

        cv2.rectangle(frame, (x,y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, model.names[int(cls)], (x, y-6), cv2.FONT_HERSHEY_PLAIN, 2, (10, 30, 255), 2)
        cv2.putText(frame, f"{conf:0.2f}", (x+70, y-6), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 2)
        
    return frame, detections