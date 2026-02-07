"""
Training script for fire detection model
"""

import argparse
import yaml
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from fire_detection.fire_detector import FireDetector
from utils.config_loader import ConfigLoader


def create_data_yaml(data_root, output_path="data/fire_detection/data.yaml"):
    """
    Create YOLO format data.yaml file.

    Args:
        data_root: Root directory of the dataset
        output_path: Path to save data.yaml
    """
    # Check which validation folder exists (valid or val)
    valid_path = Path(data_root) / 'valid'
    val_path = Path(data_root) / 'val'

    if valid_path.exists():
        val_folder = 'valid/images'
    elif val_path.exists():
        val_folder = 'val/images'
    else:
        print("Warning: Neither 'valid' nor 'val' folder found!")
        val_folder = 'valid/images'  # default

    data_config = {
        'path': str(Path(data_root).absolute()),
        'train': 'train/images',
        'val': val_folder,
        'test': 'test/images',
        'nc': 1,  # Number of classes (fire)
        'names': ['fire']
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Created data.yaml at: {output_path}")
    return str(output_path)


def train_fire_detector(config_path="config.yaml"):
    """
    Train fire detection model using configuration.

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = ConfigLoader(config_path)
    fire_config = config.get_fire_config()
    general_config = config.get_general_config()

    print("="*60)
    print("FIRE DETECTION TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {fire_config['model_name']}")
    print(f"  Image size: {fire_config['img_size']}")
    print(f"  Batch size: {fire_config['batch_size']}")
    print(f"  Epochs: {fire_config['epochs']}")
    print(f"  Device: {general_config['device']}")
    print(f"  Dataset: {fire_config['dataset']['root']}")
    print("="*60)

    # Check if dataset exists
    data_root = Path(fire_config['dataset']['root'])
    if not data_root.exists():
        print(f"\nError: Dataset directory not found: {data_root}")
        print("\nPlease download and prepare the dataset first:")
        print("  python src/utils/dataset_downloader.py")
        return

    # Create data.yaml
    data_yaml = create_data_yaml(
        data_root=data_root,
        output_path=data_root / "data.yaml"
    )

    # Initialize detector
    device = general_config['device'] if torch.cuda.is_available() else 'cpu'
    detector = FireDetector(
        conf_threshold=fire_config['confidence_threshold'],
        iou_threshold=fire_config['iou_threshold'],
        device=device
    )

    # Train model with all config parameters
    print("\nStarting training...\n")
    training_config = fire_config.get('training', {})
    results = detector.train(
        data_yaml=data_yaml,
        epochs=fire_config['epochs'],
        img_size=fire_config['img_size'],
        batch_size=fire_config['batch_size'],
        model_size=fire_config['model_name'].replace('yolov8', ''),
        lr=training_config.get('learning_rate', 0.001),
        optimizer=training_config.get('optimizer', 'Adam'),
        weight_decay=training_config.get('weight_decay', 0.0005),
        patience=training_config.get('patience', 50)
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"\nModel saved to: results/fire_detection/yolov8_fire/weights/best.pt")
    print(f"Training results: results/fire_detection/yolov8_fire/")


def main():
    parser = argparse.ArgumentParser(description='Train fire detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-yaml', type=str, default=None,
                       help='Path to data.yaml (if already created)')

    args = parser.parse_args()

    train_fire_detector(args.config)


if __name__ == "__main__":
    main()
