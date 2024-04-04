"""
Created By: ishwor subedi
Date: 2024-04-04
"""
import argparse

from ultralytics.data.annotator import auto_annotate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Annotate')
    parser.add_argument('--data_dir', type=str, default='dataset/images/', help='Path to the data')
    parser.add_argument('--det_model', type=str, default='yolo_auto_annotator/resources/yolov8x.pt',
                        help='Path to the detection model')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='', help='Device to use for detection')

    args = parser.parse_args()

    auto_annotate(data=args.data,
                  det_model=args.det_model,
                  output_dir=args.output_dir,
                  device=args.device)
