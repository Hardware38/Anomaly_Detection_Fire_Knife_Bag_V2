# Visual Anomaly Detection System - Project Overview

## Project Title
**Visual Anomaly Detection Using Deep Learning-Based Image Processing**

## Objective
Design and implement a deep learning-based visual analysis system that detects critical safety threats including:
- 🔥 Fire incidents (indoor and outdoor)
- 🚂 Individuals approaching restricted railway zones
- 🔪 Persons carrying sharp weapons

## Current Implementation Status

### ✅ Phase 1: Fire Detection (COMPLETED - Infrastructure)
The fire detection module is fully implemented and ready for training.

#### Implemented Components:

1. **YOLOv8 Fire Detector** (`src/fire_detection/fire_detector.py`)
   - Deep learning-based fire detection using state-of-the-art YOLOv8
   - Configurable model sizes (n, s, m, l, x)
   - Support for training, inference, and evaluation
   - Works on images, videos, and webcam streams

2. **Hybrid Detection System** (`src/fire_detection/hybrid_detector.py`)
   - Combines three complementary approaches:
     - **YOLOv8**: Primary deep learning detector
     - **Color-based detection**: HSV color space analysis for fire colors
     - **Motion detection**: Detects characteristic fire flickering
   - Fusion algorithm that combines all three methods
   - More robust than single-method detection

3. **Training Pipeline** (`src/fire_detection/train.py`)
   - Automated training workflow
   - Configuration-based training
   - Automatic dataset validation
   - Built-in checkpointing and metrics logging

4. **Testing Framework** (`src/fire_detection/test.py`)
   - Comprehensive evaluation on test sets
   - Performance metrics (mAP, precision, recall)
   - Support for single image, video, and batch testing
   - Hybrid system testing

5. **Real-time Demo** (`src/fire_detection/demo.py`)
   - Live webcam detection
   - Video file processing
   - Real-time FPS monitoring
   - Alert system with cooldown
   - Screenshot capability

6. **Dataset Management** (`src/utils/dataset_downloader.py`)
   - Comprehensive list of fire detection datasets
   - Download instructions for Roboflow, Kaggle, GitHub
   - Recommended datasets with quality ratings

7. **Dataset Preparation** (`src/utils/prepare_dataset.py`)
   - Converts any dataset to YOLO format
   - Automatic train/val/test splitting
   - Label validation
   - Dataset statistics

8. **Configuration System** (`config.yaml`, `src/utils/config_loader.py`)
   - Centralized configuration management
   - Easy hyperparameter tuning
   - Separate configs for each detection module

### 📋 Next Steps for Fire Detection:

1. **Download Dataset** (5-10 minutes)
   - Choose from recommended datasets in `src/utils/dataset_downloader.py`
   - Roboflow Universe has excellent pre-annotated datasets

2. **Prepare Dataset** (10-30 minutes)
   ```bash
   python src/utils/prepare_dataset.py --source path/to/downloaded/dataset
   ```

3. **Install Dependencies** (10-20 minutes)
   ```bash
   pip install -r requirements.txt
   ```

4. **Train Model** (1-5 hours depending on GPU and dataset size)
   ```bash
   python src/fire_detection/train.py
   ```

5. **Test Model**
   ```bash
   python src/fire_detection/test.py --model results/fire_detection/yolov8_fire/weights/best.pt --mode image --input test_image.jpg
   ```

6. **Real-time Demo**
   ```bash
   python src/fire_detection/demo.py --model results/fire_detection/yolov8_fire/weights/best.pt --source 0 --hybrid
   ```

### 🚧 Phase 2: Weapon Detection (NOT STARTED)
- Similar architecture to fire detection
- Requires weapon/knife detection dataset
- Will reuse much of the infrastructure

### 🚧 Phase 3: Railway Intrusion Detection (NOT STARTED)
- Person detection in restricted zones
- Zone definition and monitoring
- Will reuse detection infrastructure

### 🚧 Phase 4: Integration (NOT STARTED)
- Multi-threat monitoring system
- Unified alert system
- Dashboard for monitoring multiple cameras

## Technical Architecture

### Deep Learning Stack
- **Framework**: PyTorch 2.0+
- **Object Detection**: Ultralytics YOLOv8
- **Computer Vision**: OpenCV 4.8+
- **Data Processing**: NumPy, Pandas

### Model Architecture
```
YOLOv8 Architecture:
├── Backbone: CSPDarknet53
├── Neck: PANet
└── Head: Decoupled detection head

Hybrid System:
├── YOLOv8 Detector (60% weight)
├── Color Detector (30% weight)
└── Motion Detector (10% weight)
```

### Detection Pipeline
```
Input (Image/Video/Webcam)
    ↓
Preprocessing
    ↓
┌─────────────────┐
│  YOLOv8 Model   │ → Bounding boxes, confidence scores
└─────────────────┘
    ↓
┌─────────────────┐
│ Color Detection │ → Fire-colored regions (optional)
└─────────────────┘
    ↓
┌─────────────────┐
│Motion Detection │ → Movement patterns (optional)
└─────────────────┘
    ↓
Detection Fusion
    ↓
Alert System
    ↓
Visualization
```

## Project Structure

```
Bitirme/
├── config.yaml                    # Main configuration
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── INSTALLATION.md               # Installation guide
├── QUICKSTART.md                 # Quick start guide
├── PROJECT_OVERVIEW.md           # This file
│
├── data/                          # Dataset directory
│   ├── fire_detection/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── data.yaml
│   ├── weapon_detection/         # Future
│   └── railway_intrusion/        # Future
│
├── src/                          # Source code
│   ├── fire_detection/
│   │   ├── __init__.py
│   │   ├── fire_detector.py     # YOLOv8 detector
│   │   ├── hybrid_detector.py   # Hybrid system
│   │   ├── train.py             # Training script
│   │   ├── test.py              # Testing script
│   │   └── demo.py              # Real-time demo
│   │
│   ├── weapon_detection/         # Future
│   ├── railway_detection/        # Future
│   │
│   └── utils/
│       ├── config_loader.py     # Config management
│       ├── dataset_downloader.py # Dataset utilities
│       └── prepare_dataset.py   # Dataset preparation
│
├── models/                       # Saved model weights
├── results/                      # Training results
│   └── fire_detection/
│       └── yolov8_fire/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           └── confusion_matrix.png
│
├── notebooks/                    # Jupyter notebooks (optional)
└── logs/                        # Training logs
```

## Features

### Fire Detection Module
✅ Multiple model sizes (nano to extra-large)
✅ Hybrid detection (YOLOv8 + Color + Motion)
✅ Real-time inference
✅ Video and webcam support
✅ Configurable alert system
✅ Comprehensive evaluation metrics
✅ Easy dataset preparation
✅ Pre-trained model support

### General System Features
✅ Modular architecture (easy to extend)
✅ Comprehensive documentation
✅ Configuration-based workflow
✅ GPU acceleration support
✅ Visualization tools
✅ Alert system with cooldown
✅ FPS monitoring
✅ Screenshot capability

## Performance Expectations

### YOLOv8 Models (on typical fire detection dataset):

| Model   | Size  | mAP50 | Speed (ms) | Use Case           |
|---------|-------|-------|------------|-------------------|
| YOLOv8n | 6 MB  | ~85%  | 5-10       | Real-time, edge   |
| YOLOv8s | 22 MB | ~88%  | 10-15      | Real-time, mobile |
| YOLOv8m | 52 MB | ~91%  | 20-30      | High accuracy     |
| YOLOv8l | 87 MB | ~93%  | 30-50      | Maximum accuracy  |
| YOLOv8x | 136MB | ~94%  | 50-80      | Research          |

*Note: Performance varies based on dataset quality and size*

### Hybrid System
- Improved detection in edge cases
- Better handling of smoke vs. fire
- Reduced false positives
- ~5-10% improvement in overall accuracy

## Dataset Recommendations

### Fire Detection
**Recommended Dataset**: [Fire Detection on Roboflow](https://universe.roboflow.com/joseph-nelson/fire-detection-6xwrs)
- ~3000+ images
- Indoor and outdoor scenes
- Various fire types and sizes
- Pre-annotated in YOLO format
- Regular updates

**Alternative**: Kaggle Fire Dataset
- Free download
- Larger dataset
- May require annotation processing

## Evaluation Metrics

### Object Detection Metrics
- **mAP@0.5**: Mean Average Precision at 0.5 IoU threshold
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Real-time Performance Metrics
- **FPS**: Frames per second
- **Latency**: Detection time per frame
- **Alert accuracy**: Correct alerts / Total alerts

## Research and Documentation

### Recommended Papers
1. Vision-based fire detection using deep learning (IEEE Xplore, 2020-2025)
2. YOLO series papers (Redmon et al., Ultralytics)
3. Hybrid fire detection approaches

### Technical Documentation
- YOLOv8: https://docs.ultralytics.com/
- PyTorch: https://pytorch.org/docs/
- OpenCV: https://docs.opencv.org/

## Timeline Estimate

### Phase 1: Fire Detection
- ✅ Setup and implementation: **COMPLETED**
- ⏳ Dataset preparation: 1-2 hours
- ⏳ Training: 2-5 hours
- ⏳ Testing and evaluation: 1-2 hours
- ⏳ Documentation: 1-2 hours
- **Total**: ~1 week

### Phase 2: Weapon Detection
- Implementation: 3-4 days
- Dataset preparation: 1-2 days
- Training: 1-2 days
- Testing: 1 day
- **Total**: ~1 week

### Phase 3: Railway Intrusion
- Implementation: 3-4 days
- Dataset preparation: 1-2 days
- Training: 1-2 days
- Testing: 1 day
- **Total**: ~1 week

### Phase 4: Integration
- Multi-threat system: 2-3 days
- Dashboard: 2-3 days
- Testing: 1-2 days
- **Total**: ~1 week

**Overall Project**: 4-6 weeks

## Success Criteria

### Fire Detection Module
- [x] Implementation complete
- [ ] Model trained with mAP@0.5 > 85%
- [ ] Real-time performance (>10 FPS on GPU)
- [ ] Low false positive rate (<5%)
- [ ] Successfully detects various fire types

### Complete System
- [ ] All three detection modules implemented
- [ ] Integrated alert system
- [ ] Real-time multi-camera monitoring
- [ ] Comprehensive documentation
- [ ] IEEE paper submitted

## Current Priority: Fire Detection Training

**YOU ARE HERE** ⬇️

Next immediate actions:
1. Download fire detection dataset
2. Install dependencies
3. Prepare dataset in YOLO format
4. Train initial model
5. Evaluate and iterate

## Contact and Support

For issues or questions:
- Check documentation files
- Review code comments
- Consult YOLOv8 documentation
- Review GitHub issues for similar problems
