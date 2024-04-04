"""
Created By: ishwor subedi
Date: 2024-04-04
"""
import argparse

from src.services.object_detection_example import detect_objects

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection Example')
    parser.add_argument('--model_path', type=str,
                        default='services_trinetra/yolo_auto_annotator/resources/traffic_light.txt.pt',
                        help='Path to the model')
    parser.add_argument('--image_path', type=str, default='images/Screenshot from 2024-02-21 11-39-48.png',
                        help='Path to the image')

    args = parser.parse_args()

    detect_objects(model_path=args.model_path, image_path=args.image_path)
