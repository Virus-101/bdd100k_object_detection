"""
YOLOv8 Evaluation Script for BDD100K
=====================================
Evaluate trained model on validation set and compute metrics.

Usage:
    python evaluate.py                           # Evaluate with default settings
    python evaluate.py --weights best.pt         # Custom weights
    python evaluate.py --split val               # Evaluate on validation set
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


CLASS_NAMES = [
    'car', 'truck', 'bus', 'person', 'bike',
    'motor', 'traffic_light', 'traffic_sign', 'train', 'rider'
]


def evaluate(args):
    """Run evaluation on the validation set."""
    
    print("="*60)
    print("BDD100K OBJECT DETECTION - EVALUATION")
    print("="*60)
    print(f"\nWeights: {args.weights}")
    print(f"Data config: {args.data}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    
    # Load model
    print(f"\nLoading model...")
    model = YOLO(args.weights)
    
    # Run validation
    print(f"\nRunning evaluation...")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        save_json=args.save_json,
        project='runs/eval',
        name=f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        verbose=True,
        plots=True,
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class AP50:")
    print("-"*40)
    
    ap50_per_class = metrics.box.ap50
    if ap50_per_class is not None and len(ap50_per_class) > 0:
        for i, ap in enumerate(ap50_per_class):
            cls_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
            print(f"  {cls_name:15s}: {ap:.4f}")
    
    # Speed metrics
    print(f"\nSpeed Metrics:")
    if hasattr(metrics, 'speed'):
        print(f"  Preprocess: {metrics.speed.get('preprocess', 'N/A')} ms")
        print(f"  Inference:  {metrics.speed.get('inference', 'N/A')} ms")
        print(f"  Postprocess: {metrics.speed.get('postprocess', 'N/A')} ms")
    
    print("="*60)
    
    # Save results to JSON
    if args.save_results:
        results_dict = {
            'weights': str(args.weights),
            'data': str(args.data),
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'per_class_ap50': {
                CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}': float(ap)
                for i, ap in enumerate(ap50_per_class)
            } if ap50_per_class is not None else {},
            'timestamp': datetime.now().isoformat(),
        }
        
        results_path = Path(args.output) / 'evaluation_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv8 on BDD100K',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--weights', type=str, 
                        default='runs/train/yolov8s_bdd100k/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--data', type=str, default='configs/bdd100k.yaml',
                        help='Path to dataset config')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--save-json', action='store_true', default=False,
                        help='Save results in COCO JSON format')
    parser.add_argument('--save-results', action='store_true', default=True,
                        help='Save evaluation results to JSON')
    parser.add_argument('--output', type=str, default='runs/eval',
                        help='Output directory')
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
