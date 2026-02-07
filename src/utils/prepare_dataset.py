"""
Dataset preparation utility
Converts various dataset formats to YOLO format and splits into train/val/test
"""

import os
import shutil
from pathlib import Path
import random
import argparse
from tqdm import tqdm
import yaml


class DatasetPreparator:
    """Prepare dataset in YOLO format."""

    def __init__(self, source_dir, output_dir="data/fire_detection", class_names=None):
        """
        Initialize dataset preparator.

        Args:
            source_dir: Source dataset directory
            output_dir: Output directory for prepared dataset
            class_names: List of class names (default: ['fire'])
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names or ['fire']

        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def split_dataset(self, split_ratio=(0.7, 0.2, 0.1), seed=42):
        """
        Split dataset into train/val/test sets.

        Args:
            split_ratio: Tuple of (train, val, test) ratios
            seed: Random seed for reproducibility
        """
        print("\n" + "="*60)
        print("DATASET PREPARATION")
        print("="*60)

        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(self.source_dir.glob(f"**/*{ext}")))
            image_files.extend(list(self.source_dir.glob(f"**/*{ext.upper()}")))

        if not image_files:
            print(f"Error: No images found in {self.source_dir}")
            return

        print(f"\nFound {len(image_files)} images")

        # Filter images that have corresponding labels
        valid_pairs = []
        for img_path in tqdm(image_files, desc="Checking labels"):
            # Look for corresponding label file
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists():
                # Try in labels directory
                label_path = img_path.parent.parent / 'labels' / img_path.stem / '.txt'
                if not label_path.exists():
                    label_path = img_path.parent / 'labels' / f"{img_path.stem}.txt"

            if label_path.exists():
                valid_pairs.append((img_path, label_path))

        print(f"Found {len(valid_pairs)} valid image-label pairs")

        if not valid_pairs:
            print("Error: No valid image-label pairs found")
            print("Expected label format: YOLO format (.txt files)")
            return

        # Shuffle and split
        random.seed(seed)
        random.shuffle(valid_pairs)

        train_ratio, val_ratio, test_ratio = split_ratio
        n_train = int(len(valid_pairs) * train_ratio)
        n_val = int(len(valid_pairs) * val_ratio)

        train_pairs = valid_pairs[:n_train]
        val_pairs = valid_pairs[n_train:n_train + n_val]
        test_pairs = valid_pairs[n_train + n_val:]

        print(f"\nSplit:")
        print(f"  Train: {len(train_pairs)} ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(val_pairs)} ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(test_pairs)} ({test_ratio*100:.0f}%)")

        # Copy files
        self.copy_files(train_pairs, 'train')
        self.copy_files(val_pairs, 'val')
        self.copy_files(test_pairs, 'test')

        # Create data.yaml
        self.create_data_yaml()

        print("\n" + "="*60)
        print("DATASET PREPARATION COMPLETE!")
        print("="*60)
        print(f"\nDataset saved to: {self.output_dir}")
        print(f"Data config: {self.output_dir / 'data.yaml'}")

    def copy_files(self, pairs, split):
        """Copy image-label pairs to split directory."""
        print(f"\nCopying {split} files...")

        for img_path, label_path in tqdm(pairs):
            # Copy image
            dst_img = self.output_dir / split / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy label
            dst_label = self.output_dir / split / 'labels' / label_path.name
            shutil.copy2(label_path, dst_label)

    def create_data_yaml(self):
        """Create YOLO data.yaml configuration file."""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"\nCreated {yaml_path}")

    def validate_labels(self, check_bounds=True):
        """
        Validate YOLO format labels.

        Args:
            check_bounds: Check if bounding boxes are within valid range
        """
        print("\nValidating labels...")

        for split in ['train', 'val', 'test']:
            label_dir = self.output_dir / split / 'labels'
            label_files = list(label_dir.glob('*.txt'))

            invalid_count = 0
            for label_file in tqdm(label_files, desc=f"Validating {split}"):
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Invalid format in {label_file}:{line_num}")
                        invalid_count += 1
                        continue

                    if check_bounds:
                        try:
                            class_id, x, y, w, h = map(float, parts)
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                print(f"Out of bounds in {label_file}:{line_num}")
                                invalid_count += 1
                        except ValueError:
                            print(f"Invalid values in {label_file}:{line_num}")
                            invalid_count += 1

            if invalid_count > 0:
                print(f"  {split}: {invalid_count} invalid labels found")
            else:
                print(f"  {split}: All labels valid ✓")

    def show_statistics(self):
        """Show dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)

        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / split / 'images'
            label_dir = self.output_dir / split / 'labels'

            n_images = len(list(img_dir.glob('*')))
            n_labels = len(list(label_dir.glob('*.txt')))

            # Count objects
            total_objects = 0
            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    total_objects += len(f.readlines())

            avg_objects = total_objects / n_labels if n_labels > 0 else 0

            print(f"\n{split.upper()}:")
            print(f"  Images: {n_images}")
            print(f"  Labels: {n_labels}")
            print(f"  Total objects: {total_objects}")
            print(f"  Avg objects per image: {avg_objects:.2f}")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument('--source', type=str, required=True,
                       help='Source dataset directory')
    parser.add_argument('--output', type=str, default='data/fire_detection',
                       help='Output directory')
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.2, 0.1],
                       help='Train/val/test split ratios')
    parser.add_argument('--classes', type=str, nargs='+', default=['fire'],
                       help='Class names')
    parser.add_argument('--validate', action='store_true',
                       help='Validate labels after preparation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Prepare dataset
    preparator = DatasetPreparator(
        source_dir=args.source,
        output_dir=args.output,
        class_names=args.classes
    )

    preparator.split_dataset(
        split_ratio=tuple(args.split),
        seed=args.seed
    )

    # Validate if requested
    if args.validate:
        preparator.validate_labels()

    # Show statistics
    preparator.show_statistics()


if __name__ == "__main__":
    main()
