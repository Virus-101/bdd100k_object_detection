"""
YOLOv8 Training Script for BDD100K Object Detection
====================================================
Optimized for RTX 4050 (6GB VRAM) but configurable for any GPU.

Usage:
    python train.py                          # Train with default settings
    python train.py --model yolov8n          # Use nano model (faster, less accurate)
    python train.py --model yolov8m          # Use medium model (requires more VRAM)
    python train.py --batch 4                # Reduce batch size if OOM
    python train.py --epochs 50              # Train for 50 epochs
    python train.py --resume                 # Resume from last checkpoint
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


def check_gpu():
    """Check GPU availability and print info."""
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ CUDA is available")
        print(f"  Device: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        
        # Recommend batch size based on GPU memory
        if gpu_memory < 6:
            rec_batch = 4
        elif gpu_memory < 8:
            rec_batch = 8
        elif gpu_memory < 12:
            rec_batch = 16
        else:
            rec_batch = 32
        print(f"  Recommended batch size: {rec_batch}")
        return True, rec_batch
    else:
        print("✗ CUDA is not available - training will be slow on CPU!")
        return False, 4


def get_model_info():
    """Print available YOLOv8 model variants."""
    models = {
        'yolov8n': {'params': '3.2M', 'speed': 'fastest', 'vram': '~3GB'},
        'yolov8s': {'params': '11.2M', 'speed': 'fast', 'vram': '~4GB'},
        'yolov8m': {'params': '25.9M', 'speed': 'medium', 'vram': '~6GB'},
        'yolov8l': {'params': '43.7M', 'speed': 'slow', 'vram': '~8GB'},
        'yolov8x': {'params': '68.2M', 'speed': 'slowest', 'vram': '~10GB'},
    }
    
    print("\nAvailable YOLOv8 models:")
    print("-" * 60)
    for name, info in models.items():
        print(f"  {name}: {info['params']:>8} params | {info['speed']:>8} | VRAM: {info['vram']}")
    print("-" * 60)


def train(args):
    """Main training function."""
    
    # Check GPU
    cuda_available, rec_batch = check_gpu()
    
    # Show model options
    get_model_info()
    
    # Use recommended batch size if not specified
    batch_size = args.batch if args.batch else rec_batch
    
    # Determine model
    model_name = f"{args.model}.pt"
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {'GPU' if cuda_available else 'CPU'}")
    print(f"  Mixed precision: {args.amp}")
    print("="*60)
    
    # Verify dataset config exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n✗ Error: Dataset config not found: {args.data}")
        print("  Make sure to update the path in configs/bdd100k.yaml")
        sys.exit(1)
    
    # Load model
    if args.resume and Path(args.project).exists():
        # Find last checkpoint
        checkpoints = list(Path(args.project).glob('*/weights/last.pt'))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getmtime)
            print(f"\nResuming from: {latest}")
            model = YOLO(str(latest))
        else:
            print(f"\nNo checkpoint found, starting fresh with {model_name}")
            model = YOLO(model_name)
    else:
        print(f"\nLoading pretrained weights: {model_name}")
        model = YOLO(model_name)
    
    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_bdd100k_{timestamp}"
    
    print(f"\nStarting training...")
    print(f"Run name: {run_name}")
    print(f"TensorBoard: tensorboard --logdir {args.project}")
    print("\n" + "="*60 + "\n")
    
    # Train
    results = model.train(
        # Data
        data=args.data,
        
        # Training settings
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_size,
        
        # Device
        device=0 if cuda_available else 'cpu',
        workers=args.workers,
        
        # Project settings
        project=args.project,
        name=run_name,
        exist_ok=False,
        
        # Optimization
        amp=args.amp,  # Mixed precision
        cache=args.cache,  # Cache images in RAM
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        
        # Saving
        save=True,
        save_period=10,  # Save checkpoint every N epochs
        
        # Validation
        val=True,
        plots=True,
        
        # Other
        patience=50,  # Early stopping patience
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Print results
    print(f"\nBest weights saved to: {args.project}/{run_name}/weights/best.pt")
    print(f"Last weights saved to: {args.project}/{run_name}/weights/last.pt")
    
    # Validate
    print("\nRunning final validation...")
    metrics = model.val()
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("="*60)
    
    # Export to ONNX for deployment
    if args.export:
        print("\nExporting model to ONNX...")
        model.export(format='onnx', imgsz=args.imgsz, simplify=True)
        print("Export complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 on BDD100K',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument('--model', type=str, default='yolov8s',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 model variant')
    
    # Data
    parser.add_argument('--data', type=str, default='configs/bdd100k.yaml',
                        help='Path to dataset YAML config')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (auto-detected if not specified)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Optimization
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='Disable mixed precision')
    parser.add_argument('--cache', action='store_true', default=False,
                        help='Cache images in RAM (faster but uses more memory)')
    
    # Project
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory for saving runs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    
    # Export
    parser.add_argument('--export', action='store_true', default=False,
                        help='Export to ONNX after training')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BDD100K OBJECT DETECTION - YOLOv8 TRAINING")
    print("="*60)
    
    train(args)


if __name__ == '__main__':
    main()
