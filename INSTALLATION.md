# Installation Guide

Complete installation guide for the Visual Anomaly Detection System.

## System Requirements

### Hardware Requirements
- **GPU (Recommended)**: NVIDIA GPU with at least 4GB VRAM
  - For YOLOv8n: 4GB VRAM
  - For YOLOv8m/l: 8GB+ VRAM
- **CPU**: Multi-core processor (Intel i5/i7 or AMD Ryzen 5/7)
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: At least 10GB free space for datasets and models

### Software Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS
- **Python**: 3.8 or higher (3.9 or 3.10 recommended)
- **CUDA**: 11.7+ (for GPU support)
- **cuDNN**: Compatible with your CUDA version

## Step-by-Step Installation

### 1. Install Python

#### Windows:
Download and install from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"

#### Linux:
```bash
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv
```

#### macOS:
```bash
brew install python@3.9
```

Verify installation:
```bash
python --version
# Should show Python 3.8 or higher
```

### 2. Install CUDA and cuDNN (For GPU Support)

#### Check if you have NVIDIA GPU:
```bash
nvidia-smi
```

If you see GPU information, proceed with CUDA installation.

#### Windows/Linux:
1. Download CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Install CUDA Toolkit 11.7 or later
3. Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
4. Extract and copy cuDNN files to CUDA installation directory

#### Verify CUDA:
```bash
nvcc --version
```

### 3. Clone or Download Project

```bash
cd /path/to/your/projects
# If using git
git clone <repository-url>
cd Bitirme

# Or simply navigate to the project directory
cd Bitirme
```

### 4. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 5. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (GPU)
# For CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (if no GPU)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### 6. Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Test Ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics YOLOv8 installed successfully')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

Expected output:
```
PyTorch: 2.0.0+cu117
CUDA available: True
CUDA version: 11.7
Ultralytics YOLOv8 installed successfully
OpenCV: 4.8.0
```

### 7. Test Fire Detector Module

```bash
python src/fire_detection/fire_detector.py
```

Expected output:
```
Fire Detector initialized
Device: cuda
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size in `config.yaml`
- Use smaller model (yolov8n instead of yolov8m)
- Close other GPU-intensive applications
- Use CPU if GPU memory is insufficient

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue: "ImportError: DLL load failed" (Windows)
**Solution:**
- Install Microsoft Visual C++ Redistributable
- Reinstall PyTorch
- Check CUDA and cuDNN installation

### Issue: "CUDA not available" but you have NVIDIA GPU
**Solution:**
1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```
3. Check CUDA version compatibility

### Issue: OpenCV error "cv2.imshow() not working"
**Solution:**
- Linux: Install required packages
  ```bash
  sudo apt install libgl1-mesa-glx
  sudo apt install libglib2.0-0
  ```
- For headless servers, use `--no-display` flag

### Issue: "Permission denied" errors
**Solution:**
```bash
# Linux/macOS
sudo chown -R $USER:$USER .
chmod +x src/fire_detection/*.py

# Windows: Run as administrator
```

## Performance Optimization

### For Training:
1. Use GPU if available
2. Increase batch size (if you have enough VRAM)
3. Use mixed precision training (automatic in YOLOv8)
4. Use multiple workers for data loading

### For Inference:
1. Use TensorRT for faster inference (advanced)
2. Use smaller model for real-time applications
3. Reduce input image size if speed is critical
4. Use model quantization (INT8) for edge devices

## Optional: Install Additional Tools

### Kaggle CLI (for downloading datasets)
```bash
pip install kaggle

# Setup Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings/account
# 2. Create New API Token
# 3. Place kaggle.json in:
#    - Linux/Mac: ~/.kaggle/
#    - Windows: C:\Users\<username>\.kaggle\
```

### Weights & Biases (for experiment tracking)
```bash
pip install wandb
wandb login
```

### Jupyter Notebook (for experiments)
```bash
pip install jupyter
jupyter notebook
```

## Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade ultralytics
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows

# Remove downloaded models and results (optional)
rm -rf results/ models/
```

## Next Steps

After successful installation:
1. Read [QUICKSTART.md](QUICKSTART.md) to get started
2. Download fire detection dataset
3. Train your first model
4. Test on images/videos

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify all dependencies are correctly installed
3. Check CUDA/GPU compatibility
4. Review error messages carefully
5. Search for similar issues online

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Installation](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html)
