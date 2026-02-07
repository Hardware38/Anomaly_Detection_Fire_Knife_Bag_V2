# Fire Detection Quick Start Guide

This guide will help you get started with the fire detection system.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (Deep Learning Framework)
- Ultralytics YOLOv8 (Object Detection)
- OpenCV (Computer Vision)
- And other required libraries

## Step 2: Download Fire Detection Dataset

You have several options:

### Option A: Using Roboflow (Recommended)

1. Visit one of these datasets:
   - [Fire Detection by Joseph Nelson](https://universe.roboflow.com/joseph-nelson/fire-detection-6xwrs)
   - [Fire Detection System](https://universe.roboflow.com/roboflow-universe-projects/fire-detection-system)

2. Click "Download" and select "YOLOv8" format

3. Extract the downloaded dataset to `data/fire_detection/`

4. The structure should be:
   ```
   data/fire_detection/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

### Option B: Using Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (get kaggle.json from https://www.kaggle.com/settings)
# Place it in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d phylake1337/fire-dataset
```

### Option C: List All Available Datasets

```bash
python src/utils/dataset_downloader.py
```

This will show you all recommended datasets and download instructions.

## Step 3: Verify Dataset Structure

Make sure your dataset follows YOLO format:
- Images in `images/` folder
- Labels in `labels/` folder (one .txt file per image)
- Each label file contains: `class_id center_x center_y width height` (normalized 0-1)

## Step 4: Configure Training

Edit `config.yaml` if needed:

```yaml
fire_detection:
  model_name: "yolov8n"  # Options: n, s, m, l, x (larger = more accurate but slower)
  img_size: 640
  batch_size: 16
  epochs: 100
  confidence_threshold: 0.5
```

## Step 5: Train the Model

```bash
python src/fire_detection/train.py
```

Training will:
- Create a `data.yaml` file automatically
- Train YOLOv8 on your dataset
- Save the best model to `results/fire_detection/yolov8_fire/weights/best.pt`
- Generate training plots and metrics

Expected training time:
- YOLOv8n: ~1-2 hours (1000 images, GPU)
- YOLOv8s: ~2-3 hours
- YOLOv8m: ~3-5 hours

## Step 6: Test the Model

### Test on a single image:

```bash
python src/fire_detection/test.py \
  --model results/fire_detection/yolov8_fire/weights/best.pt \
  --mode image \
  --input path/to/test/image.jpg \
  --output results/output.jpg
```

### Test on video:

```bash
python src/fire_detection/test.py \
  --model results/fire_detection/yolov8_fire/weights/best.pt \
  --mode video \
  --input path/to/test/video.mp4 \
  --output results/output.mp4
```

### Test with Hybrid System:

```bash
python src/fire_detection/test.py \
  --model results/fire_detection/yolov8_fire/weights/best.pt \
  --mode image \
  --input path/to/test/image.jpg \
  --hybrid
```

The hybrid system combines:
- YOLOv8 deep learning detection
- Color-based fire detection (HSV color space)
- Motion detection (for video/webcam)

### Evaluate on test dataset:

```bash
python src/fire_detection/test.py \
  --model results/fire_detection/yolov8_fire/weights/best.pt \
  --mode dataset \
  --data-yaml data/fire_detection/data.yaml
```

## Step 7: Real-time Webcam Detection (Coming Soon)

```bash
python src/fire_detection/demo.py \
  --model results/fire_detection/yolov8_fire/weights/best.pt \
  --source 0 \
  --hybrid
```

## Project Structure

```
Bitirme/
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── data/
│   └── fire_detection/           # Dataset directory
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── fire_detection/
│   │   ├── fire_detector.py     # YOLOv8 fire detector
│   │   ├── hybrid_detector.py   # Hybrid detection system
│   │   ├── train.py             # Training script
│   │   └── test.py              # Testing script
│   └── utils/
│       ├── config_loader.py     # Configuration loader
│       └── dataset_downloader.py # Dataset downloader
├── models/                       # Saved models
└── results/                      # Training results and outputs
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.yaml` (try 8 or 4)
- Use a smaller model (yolov8n instead of yolov8m)

### Dataset Not Found
- Verify dataset structure matches YOLO format
- Check paths in `config.yaml`

### Poor Detection Performance
- Train for more epochs (increase to 150-200)
- Use a larger model (yolov8m or yolov8l)
- Augment your dataset with more diverse fire images
- Adjust confidence threshold in config

### Slow Inference
- Use GPU if available
- Use smaller model (yolov8n or yolov8s)
- Reduce image size in config

## Next Steps

1. Collect more diverse fire images to improve robustness
2. Implement weapon detection module
3. Implement railway intrusion detection module
4. Build integrated multi-threat detection system
5. Deploy as real-time monitoring system

## Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- Roboflow Datasets: https://universe.roboflow.com/
- PyTorch Documentation: https://pytorch.org/docs/
- OpenCV Documentation: https://docs.opencv.org/
