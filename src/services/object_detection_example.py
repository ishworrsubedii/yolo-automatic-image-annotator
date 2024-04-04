"""
Created By: ishwor subedi
Date: 2024-02-21
"""

import cv2
from ultralytics import YOLO


def detect_objects(model_path, image_path):
    """
    This function takes in the path to a directory containing images and automatically annotates the images using the
    :param model_path: Path to the detection model (cpu,cuda)
    :param image_path: Path to the directory containing images
    :return:
    """
    img = cv2.imread(image_path)
    obj = YOLO(model_path)

    result = obj.predict(img, conf=0.5, iou=0.4)

    detected_objects = result[0].boxes[0]

    for bbox in detected_objects:
        label = result[0].names[bbox[0]]
        confidence = bbox[1]
        box = bbox[2:]

        x1, y1, x2, y2 = box

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(img, f'{label}: {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

    cv2.imshow('Annotated Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


