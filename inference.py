"""
YOLOv8 Inference Script for BDD100K Object Detection
=====================================================
Run inference on images, videos, or webcam using trained model.

Usage:
    python inference.py --source image.jpg                    # Single image
    python inference.py --source images/                      # Folder of images
    python inference.py --source video.mp4                    # Video file
    python inference.py --source 0                            # Webcam
    python inference.py --source image.jpg --weights best.pt  # Custom weights
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# Class names for BDD100K
CLASS_NAMES = [
    'car', 'truck', 'bus', 'person', 'bike',
    'motor', 'traffic_light', 'traffic_sign', 'train', 'rider'
]

# Colors for each class (BGR format for OpenCV)
CLASS_COLORS = [
    (0, 0, 255),      # car - red
    (0, 255, 0),      # truck - green
    (255, 0, 0),      # bus - blue
    (0, 255, 255),    # person - yellow
    (255, 0, 255),    # bike - magenta
    (255, 255, 0),    # motor - cyan
    (0, 128, 255),    # traffic_light - orange
    (255, 0, 128),    # traffic_sign - purple
    (128, 128, 128),  # train - gray
    (203, 192, 255),  # rider - pink
]


def load_model(weights_path: str) -> YOLO:
    """Load YOLO model from weights file."""
    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    return model


def run_inference(
    model: YOLO,
    source: str,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    save: bool = True,
    show: bool = False,
    output_dir: str = 'runs/inference'
):
    """
    Run inference on source.
    
    Args:
        model: Loaded YOLO model
        source: Path to image/video/folder or webcam ID
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold for NMS
        save: Save results to output_dir
        show: Display results in window
        output_dir: Directory to save results
    """
    print(f"\nRunning inference on: {source}")
    print(f"Confidence threshold: {conf_thresh}")
    print(f"IoU threshold: {iou_thresh}")
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf_thresh,
        iou=iou_thresh,
        save=save,
        show=show,
        project=output_dir,
        name='predict',
        exist_ok=True,
        verbose=True,
    )
    
    return results


def process_results(results, print_details: bool = True):
    """Process and print detection results."""
    total_detections = 0
    class_counts = {}
    
    for result in results:
        boxes = result.boxes
        
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                total_detections += 1
                
                if print_details:
                    print(f"  {cls_name:15s} | conf: {conf:.3f} | box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
    
    print(f"\nTotal detections: {total_detections}")
    print("Class breakdown:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls_name:15s}: {count}")
    
    return class_counts


def inference_single_image(
    model: YOLO,
    image_path: str,
    conf_thresh: float = 0.25,
    save_path: str = None
) -> np.ndarray:
    """
    Run inference on a single image and return annotated result.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf_thresh: Confidence threshold
        save_path: Optional path to save annotated image
    
    Returns:
        Annotated image as numpy array
    """
    # Run inference
    results = model.predict(image_path, conf=conf_thresh, verbose=False)
    
    # Get annotated image
    annotated = results[0].plot()
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved annotated image to: {save_path}")
    
    return annotated


def realtime_webcam(model: YOLO, conf_thresh: float = 0.25, camera_id: int = 0):
    """
    Run real-time inference on webcam.
    
    Args:
        model: Loaded YOLO model
        conf_thresh: Confidence threshold
        camera_id: Webcam device ID
    """
    print(f"\nStarting webcam inference (camera {camera_id})")
    print("Press 'q' to quit, 's' to save frame")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        
        # Get annotated frame
        annotated = results[0].plot()
        
        # Add FPS counter
        cv2.putText(annotated, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLOv8 Object Detection', annotated)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f'frame_{frame_count}.jpg'
            cv2.imwrite(save_path, annotated)
            print(f"Saved: {save_path}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 Inference for BDD100K',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--source', type=str, required=True,
                        help='Image/video path, folder, or webcam ID (0, 1, ...)')
    parser.add_argument('--weights', type=str, default='runs/train/yolov8s_bdd100k/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results')
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help='Do not save results')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Display results')
    parser.add_argument('--output', type=str, default='runs/inference',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BDD100K OBJECT DETECTION - YOLOv8 INFERENCE")
    print("="*60)
    
    # Load model
    model = load_model(args.weights)
    
    # Check if source is webcam
    if args.source.isdigit():
        realtime_webcam(model, args.conf, int(args.source))
    else:
        # Run inference
        results = run_inference(
            model=model,
            source=args.source,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            save=args.save,
            show=args.show,
            output_dir=args.output
        )
        
        # Process results
        process_results(results)
        
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
