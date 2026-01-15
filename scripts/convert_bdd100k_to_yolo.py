"""
BDD100K to YOLO Format Converter
================================
Converts BDD100K annotations to YOLO format for training with Ultralytics YOLOv8.

Usage:
    python convert_bdd100k_to_yolo.py --data_dir /path/to/bdd100k --output_dir /path/to/output

Expected input structure:
    bdd100k/
    ├── images/
    │   └── 100k/
    │       ├── train/
    │       ├── val/
    │       └── test/
    └── labels/
        ├── bdd100k_labels_images_train.json
        └── bdd100k_labels_images_val.json

Output structure:
    output_dir/
    ├── images/
    │   ├── train/ -> symlink to bdd100k images
    │   └── val/   -> symlink to bdd100k images
    └── labels/
        ├── train/
        └── val/
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# BDD100K class mapping to YOLO indices
CLASSES = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'person': 3,
    'bike': 4,
    'motor': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'train': 8,
    'rider': 9
}

# Reverse mapping for reference
CLASS_NAMES = {v: k for k, v in CLASSES.items()}

# BDD100K image dimensions
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def convert_bbox_to_yolo(box: dict, img_width: int = IMG_WIDTH, img_height: int = IMG_HEIGHT) -> tuple:
    """
    Convert BDD100K bounding box format to YOLO format.
    
    BDD100K format: x1, y1, x2, y2 (absolute coordinates)
    YOLO format: x_center, y_center, width, height (normalized 0-1)
    
    Args:
        box: Dictionary with x1, y1, x2, y2 keys
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to 0-1
    """
    x1 = box['x1']
    y1 = box['y1']
    x2 = box['x2']
    y2 = box['y2']
    
    # Clamp coordinates to image bounds
    x1 = max(0, min(x1, img_width))
    x2 = max(0, min(x2, img_width))
    y1 = max(0, min(y1, img_height))
    y2 = max(0, min(y2, img_height))
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height


def process_single_image(item: dict, labels_dir: Path) -> dict:
    """
    Process annotations for a single image.
    
    Args:
        item: Single image annotation from BDD100K
        labels_dir: Output directory for label files
    
    Returns:
        Statistics dictionary
    """
    stats = {'total_boxes': 0, 'skipped_boxes': 0, 'class_counts': {}}
    
    image_name = item['name']
    labels = item.get('labels', [])
    
    # Output label file (same name as image, .txt extension)
    label_file = labels_dir / image_name.replace('.jpg', '.txt')
    
    lines = []
    for label in labels:
        category = label.get('category', '')
        
        # Skip unknown categories
        if category not in CLASSES:
            stats['skipped_boxes'] += 1
            continue
        
        # Skip if no bounding box
        if 'box2d' not in label:
            stats['skipped_boxes'] += 1
            continue
        
        box = label['box2d']
        class_id = CLASSES[category]
        
        # Convert to YOLO format
        x, y, w, h = convert_bbox_to_yolo(box)
        
        # Skip invalid boxes (too small or invalid dimensions)
        if w <= 0 or h <= 0 or w > 1 or h > 1:
            stats['skipped_boxes'] += 1
            continue
        
        lines.append(f'{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
        stats['total_boxes'] += 1
        stats['class_counts'][category] = stats['class_counts'].get(category, 0) + 1
    
    # Write label file
    with open(label_file, 'w') as f:
        f.write('\n'.join(lines))
    
    return stats


def convert_split(data_dir: Path, output_dir: Path, split: str = 'train', num_workers: int = 8) -> dict:
    """
    Convert a single split (train/val) of BDD100K to YOLO format.
    
    Args:
        data_dir: Path to BDD100K dataset root
        output_dir: Path to output directory
        split: 'train' or 'val'
        num_workers: Number of parallel workers
    
    Returns:
        Statistics dictionary for the split
    """
    # Determine label file path
    if split == 'train':
        json_file = data_dir / 'labels' / 'bdd100k_labels_images_train.json'
    else:
        json_file = data_dir / 'labels' / 'bdd100k_labels_images_val.json'
    
    if not json_file.exists():
        # Try alternative path
        json_file = data_dir / 'labels' / f'det_{split}.json'
    
    if not json_file.exists():
        print(f"Warning: Label file not found: {json_file}")
        return {}
    
    print(f"\nLoading {split} annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} images in {split} set")
    
    # Create output directory
    labels_dir = output_dir / 'labels' / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images in parallel
    total_stats = {'total_boxes': 0, 'skipped_boxes': 0, 'class_counts': {}, 'images': len(data)}
    
    print(f"Converting {split} annotations...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, item, labels_dir): item for item in data}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'{split}'):
            stats = future.result()
            total_stats['total_boxes'] += stats['total_boxes']
            total_stats['skipped_boxes'] += stats['skipped_boxes']
            for cls, count in stats['class_counts'].items():
                total_stats['class_counts'][cls] = total_stats['class_counts'].get(cls, 0) + count
    
    return total_stats


def create_symlinks(data_dir: Path, output_dir: Path):
    """
    Create symlinks for images to avoid copying.
    
    Args:
        data_dir: Path to BDD100K dataset root
        output_dir: Path to output directory
    """
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
        source = data_dir / 'images' / '100k' / split
        target = images_dir / split
        
        if source.exists() and not target.exists():
            os.symlink(source.absolute(), target)
            print(f"Created symlink: {target} -> {source}")
        elif not source.exists():
            print(f"Warning: Source images not found: {source}")


def print_statistics(stats: dict, split: str):
    """Print conversion statistics."""
    print(f"\n{'='*50}")
    print(f"{split.upper()} SPLIT STATISTICS")
    print(f"{'='*50}")
    print(f"Total images: {stats.get('images', 0):,}")
    print(f"Total bounding boxes: {stats.get('total_boxes', 0):,}")
    print(f"Skipped boxes: {stats.get('skipped_boxes', 0):,}")
    print(f"\nClass distribution:")
    
    class_counts = stats.get('class_counts', {})
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        percentage = count / stats['total_boxes'] * 100 if stats['total_boxes'] > 0 else 0
        print(f"  {cls_name:15s}: {count:8,} ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Convert BDD100K to YOLO format')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BDD100K dataset root')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--no_symlinks', action='store_true',
                        help='Skip creating image symlinks')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print(f"BDD100K to YOLO Converter")
    print(f"{'='*50}")
    print(f"Input directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    
    # Check input directory
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert train and val splits
    train_stats = convert_split(data_dir, output_dir, 'train', args.workers)
    val_stats = convert_split(data_dir, output_dir, 'val', args.workers)
    
    # Print statistics
    if train_stats:
        print_statistics(train_stats, 'train')
    if val_stats:
        print_statistics(val_stats, 'val')
    
    # Create symlinks for images
    if not args.no_symlinks:
        print("\nCreating image symlinks...")
        create_symlinks(data_dir, output_dir)
    
    print(f"\n{'='*50}")
    print("CONVERSION COMPLETE!")
    print(f"{'='*50}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Update the 'path' in configs/bdd100k.yaml to point to: {output_dir}")
    print(f"2. Run: python train.py")


if __name__ == '__main__':
    main()
