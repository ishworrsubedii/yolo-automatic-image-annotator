"""
Created By: ishwor subedi
Date: 2024-02-06
"""
import os
from pathlib import Path
from ultralytics import SAM, YOLO


def auto_annotate(data, det_model="resources/yolov8x.pt", device="", output_dir=None):
    classes = [0, 1, 2, 3, 5, 7]

    det_model = YOLO(det_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device)

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        vehicle_detections = [i for i, cls_id in enumerate(class_ids) if cls_id in classes]

        if len(vehicle_detections):
            boxes = result.boxes.xyxy

            image_array = result.orig_img

            image_height, image_width, _ = image_array.shape
            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in vehicle_detections:
                    x_min, y_min, x_max, y_max = boxes[i].tolist()
                    center_x = ((x_min + x_max) / 2) / image_width
                    center_y = ((y_min + y_max) / 2) / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    f.write(f"{class_ids[i]} {center_x} {center_y} {width} {height}\n")
                    print(f"{class_ids[i]} {center_x} {center_y} {width} {height}")
        else:
            os.remove(result.path)
