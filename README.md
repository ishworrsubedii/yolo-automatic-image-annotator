# YOLO Auto Annotation

Simple script to automatically annotate images using YOLOv8.

## Requirements

- Python 3.10
- Ultralytics YOLOv8
- OpenCV

## Usage

1. Clone the repository

```angular2html
git clone https://github.com/ishworrsubedii/yolo-automatic-image-annotator.git
```

2. Install the requirements

```angular2html
pip install -r requirements.txt

```

3. Run the script

```angular2html
python examples/auto_annotate_save_annotation_example.py --data_dir dataset/images/ --det_model resources/yolov8s.pt --output_dir /dataset/detected_extracted/ --device cuda
```
