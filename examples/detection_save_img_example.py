"""
Created By: ishwor subedi
Date: 2024-04-04
"""
import argparse

from src.services.detection_save_images import detection_save_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Auto Annotation')
    parser.add_argument('--data', type=str, default='dataset/images/', help='Path to the data')
    parser.add_argument('--det_model', type=str, default='resources/yolov8s.pt', help='Path to the detection model')
    parser.add_argument('--output_dir', type=str, default='/dataset/detected_extracted/', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for detection')

    args = parser.parse_args()

    detection_save_img(data=args.data,
                       det_model=args.det_model,
                       output_dir=args.output_dir,
                       device=args.device)
