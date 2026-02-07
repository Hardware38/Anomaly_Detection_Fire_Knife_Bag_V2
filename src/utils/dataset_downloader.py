"""
Dataset Downloader for Fire Detection
Downloads and prepares fire detection datasets from various sources
"""

import os
import requests
from pathlib import Path
import zipfile
import shutil


class FireDatasetDownloader:
    """Download fire detection datasets."""

    def __init__(self, data_root="data/fire_detection"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def download_from_roboflow(self, api_key=None, workspace=None, project=None, version=1):
        """
        Download dataset from Roboflow.

        Instructions:
        1. Go to https://roboflow.com/
        2. Search for "fire detection" datasets
        3. Popular datasets:
           - Fire Detection Computer Vision Project
           - Fire and Smoke Detection
        4. Click on the dataset and get API key from export section
        """
        print("To download from Roboflow:")
        print("1. Visit: https://universe.roboflow.com/")
        print("2. Search for 'fire detection' or 'fire smoke detection'")
        print("3. Popular datasets:")
        print("   - https://universe.roboflow.com/joseph-nelson/fire-detection-6xwrs")
        print("   - https://universe.roboflow.com/roboflow-universe-projects/fire-detection-system")
        print("4. Get API key from the export section (YOLO format)")
        print("\nAlternatively, use the automated download method below.")

    def download_kaggle_dataset(self, dataset_name):
        """
        Download dataset from Kaggle.

        Popular fire detection datasets on Kaggle:
        - 'phylake1337/fire-dataset'
        - 'kutaykutlu/fire-dataset'
        - 'atulyakumar98/test-dataset'

        Requirements:
        1. Install kaggle: pip install kaggle
        2. Setup Kaggle API credentials: https://github.com/Kaggle/kaggle-api
        """
        try:
            import kaggle
            print(f"Downloading {dataset_name} from Kaggle...")
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(self.data_root / "raw"),
                unzip=True
            )
            print(f"Dataset downloaded to: {self.data_root / 'raw'}")
        except ImportError:
            print("Error: kaggle package not installed.")
            print("Install with: pip install kaggle")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nSetup Kaggle API:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Create New API Token")
            print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")

    def list_recommended_datasets(self):
        """List recommended fire detection datasets."""
        print("\n=== RECOMMENDED FIRE DETECTION DATASETS ===\n")

        print("1. ROBOFLOW UNIVERSE:")
        print("   - Fire Detection (Joseph Nelson): https://universe.roboflow.com/joseph-nelson/fire-detection-6xwrs")
        print("   - Fire and Smoke Detection: https://universe.roboflow.com/roboflow-universe-projects/fire-detection-system")
        print("   - Format: YOLOv8 compatible")
        print("   - Size: ~1000-5000 images")
        print()

        print("2. KAGGLE DATASETS:")
        print("   - Fire Dataset (phylake1337): https://www.kaggle.com/datasets/phylake1337/fire-dataset")
        print("     kaggle datasets download -d phylake1337/fire-dataset")
        print()
        print("   - Fire Dataset (kutaykutlu): https://www.kaggle.com/datasets/kutaykutlu/fire-dataset")
        print("     kaggle datasets download -d kutaykutlu/fire-dataset")
        print()
        print("   - FIRE Dataset: https://www.kaggle.com/datasets/atulyakumar98/test-dataset")
        print("     kaggle datasets download -d atulyakumar98/test-dataset")
        print()

        print("3. GITHUB DATASETS:")
        print("   - DeepQuest AI Fire Dataset: https://github.com/DeepQuestAI/Fire-Smoke-Dataset")
        print("   - Fire Detection Dataset: https://github.com/cair/Fire-Detection-Image-Dataset")
        print()

        print("4. ACADEMIC DATASETS:")
        print("   - BoWFire Dataset: https://github.com/steffensbola/bowfire-dataset")
        print("   - FiSmo (Fire and Smoke): https://github.com/hhzzxx957/FiSmo")
        print()

        print("=== DOWNLOAD INSTRUCTIONS ===\n")
        print("Option A - Using this script:")
        print("  downloader = FireDatasetDownloader()")
        print("  downloader.download_kaggle_dataset('phylake1337/fire-dataset')")
        print()
        print("Option B - Manual download:")
        print("  1. Visit the dataset URL")
        print("  2. Download and extract to: data/fire_detection/raw/")
        print("  3. Run dataset preparation script")
        print()

    def prepare_yolo_format(self, source_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
        """
        Prepare dataset in YOLO format with train/val/test split.

        Expected source structure:
        source_dir/
            images/
            labels/
        """
        print(f"Preparing YOLO format dataset...")
        print(f"Source: {source_dir}")
        print(f"Output: {output_dir}")
        print(f"Split ratio - Train: {split_ratio[0]}, Val: {split_ratio[1]}, Test: {split_ratio[2]}")

        # This will be implemented after we get the actual dataset
        print("\nNote: Dataset preparation will be customized based on downloaded dataset structure.")


def main():
    """Main function to demonstrate usage."""
    downloader = FireDatasetDownloader()
    downloader.list_recommended_datasets()

    print("\n=== NEXT STEPS ===")
    print("1. Choose a dataset from the list above")
    print("2. Download it manually or use the download methods")
    print("3. Run dataset preparation to convert to YOLO format")
    print("4. Start training the fire detection model")


if __name__ == "__main__":
    main()
