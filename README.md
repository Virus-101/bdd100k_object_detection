# BDD100K Object Detection with YOLOv8

Complete training pipeline for object detection on the BDD100K dataset using Ultralytics YOLOv8, optimized for RTX 4050 (6GB VRAM).

## Project Structure

```
bdd100k_object_detection/
├── configs/
│   └── bdd100k.yaml          # Dataset configuration
├── scripts/
│   └── convert_bdd100k_to_yolo.py  # BDD100K to YOLO converter
├── train.py                   # Training script
├── inference.py               # Inference script
├── evaluate.py                # Evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### Step 1: Install Dependencies

```bash
# Create conda environment (recommended)
conda create -n bdd100k python=3.10 -y
conda activate bdd100k

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Step 2: Download BDD100K Dataset

1. **Register** at https://bdd-data.berkeley.edu/
2. **Download** from https://bdd-data.berkeley.edu/portal.html:
   - `bdd100k_images_100k.zip` (~7GB)
   - `bdd100k_labels_release.zip` (~52MB)

3. **Extract** to a folder:
```bash
mkdir -p ~/datasets/bdd100k
cd ~/datasets/bdd100k
unzip bdd100k_images_100k.zip
unzip bdd100k_labels_release.zip
```

Expected structure:
```
bdd100k/
├── images/
│   └── 100k/
│       ├── train/  (70,000 images)
│       ├── val/    (10,000 images)
│       └── test/   (20,000 images)
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

### Step 3: Convert to YOLO Format

```bash
python scripts/convert_bdd100k_to_yolo.py \
    --data_dir ~/datasets/bdd100k \
    --output_dir ~/datasets/bdd100k_yolo
```

This creates:
```
bdd100k_yolo/
├── images/
│   ├── train/ -> symlink
│   └── val/   -> symlink
└── labels/
    ├── train/  (70,000 .txt files)
    └── val/    (10,000 .txt files)
```

### Step 4: Update Dataset Config

Edit `configs/bdd100k.yaml` and set the correct path:

```yaml
path: /home/YOUR_USERNAME/datasets/bdd100k_yolo  # UPDATE THIS!
```

### Step 5: Train

```bash
# Default training (YOLOv8s, 100 epochs)
python train.py

# Or with custom settings
python train.py --model yolov8n --epochs 50 --batch 8
```

### Step 6: Monitor Training

```bash
# In another terminal
tensorboard --logdir runs/train
```

Open http://localhost:6006 in your browser.

### Step 7: Evaluate

```bash
python evaluate.py --weights runs/train/yolov8s_bdd100k_*/weights/best.pt
```

### Step 8: Run Inference

```bash
# On an image
python inference.py --source test_image.jpg --weights runs/train/.../weights/best.pt

# On a video
python inference.py --source driving_video.mp4

# On webcam
python inference.py --source 0
```

---

## Dataset Information

| Property | Value |
|----------|-------|
| Total Images | 100,000 |
| Training | 70,000 |
| Validation | 10,000 |
| Test | 20,000 |
| Resolution | 1280 × 720 |
| Classes | 10 |

### Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | car | Passenger vehicles |
| 1 | truck | Commercial trucks |
| 2 | bus | Buses |
| 3 | person | Pedestrians |
| 4 | bike | Bicycles |
| 5 | motor | Motorcycles |
| 6 | traffic_light | Traffic lights |
| 7 | traffic_sign | Traffic signs |
| 8 | train | Trains |
| 9 | rider | Cyclists/motorcyclists |

---

## Training Options

### Model Selection

| Model | Parameters | Speed | VRAM Required | Recommended For |
|-------|------------|-------|---------------|-----------------|
| yolov8n | 3.2M | Fastest | ~3GB | Testing, edge devices |
| yolov8s | 11.2M | Fast | ~4GB | **RTX 4050 (default)** |
| yolov8m | 25.9M | Medium | ~6GB | Better accuracy |
| yolov8l | 43.7M | Slow | ~8GB | High accuracy |
| yolov8x | 68.2M | Slowest | ~10GB | Maximum accuracy |

### Command Line Arguments

```bash
python train.py --help

# Key options:
--model      # Model variant (yolov8n/s/m/l/x)
--epochs     # Number of epochs (default: 100)
--batch      # Batch size (auto-detected if not set)
--imgsz      # Image size (default: 640)
--amp        # Mixed precision (default: enabled)
--cache      # Cache images in RAM (faster but uses more memory)
--resume     # Resume from last checkpoint
```

### Example Commands

```bash
# Quick test (nano model, few epochs)
python train.py --model yolov8n --epochs 10 --batch 16

# Standard training for RTX 4050
python train.py --model yolov8s --epochs 100 --batch 8

# High-accuracy training (if you have more VRAM)
python train.py --model yolov8m --epochs 150 --batch 4

# Resume interrupted training
python train.py --resume
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
python train.py --batch 4

# Use smaller model
python train.py --model yolov8n

# Reduce image size
python train.py --imgsz 480
```

### Slow Training

```bash
# Enable image caching (needs RAM)
python train.py --cache

# Increase workers
python train.py --workers 8
```

### CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Dataset Not Found

Make sure to:
1. Update the `path` in `configs/bdd100k.yaml`
2. Run the conversion script first
3. Check that symlinks were created correctly

---

## Expected Results

After 100 epochs on BDD100K with YOLOv8s:

| Metric | Expected Value |
|--------|---------------|
| mAP50 | 0.55 - 0.65 |
| mAP50-95 | 0.35 - 0.45 |
| Precision | 0.60 - 0.70 |
| Recall | 0.50 - 0.60 |

Training time on RTX 4050:
- YOLOv8n: ~8-12 hours
- YOLOv8s: ~12-18 hours
- YOLOv8m: ~20-30 hours

---

## Export for Deployment

After training, export to different formats:

```python
from ultralytics import YOLO

model = YOLO('runs/train/.../weights/best.pt')

# Export formats
model.export(format='onnx')       # ONNX (recommended)
model.export(format='torchscript') # TorchScript
model.export(format='engine')      # TensorRT (fastest inference)
model.export(format='openvino')    # OpenVINO
```

---

## Resources

- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

---

## License

This project is for educational purposes. BDD100K dataset has its own license - check the official website for terms.
