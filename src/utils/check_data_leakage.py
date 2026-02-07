"""
Data Leakage Detection Utility
Checks for duplicate images across train/val/test splits
"""

import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse


def compute_image_hash(image_path):
    """Compute MD5 hash of image file."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def check_data_leakage(dataset_root):
    """
    Check for data leakage between train/val/test splits.

    Args:
        dataset_root: Root directory of dataset
    """
    dataset_root = Path(dataset_root)

    print("="*60)
    print("DATA LEAKAGE DETECTION")
    print("="*60)

    # Collect all images from each split
    splits = {
        'train': list((dataset_root / 'train' / 'images').glob('*.[jp][pn]g')),
        'val': list((dataset_root / 'val' / 'images').glob('*.[jp][pn]g')),
        'valid': list((dataset_root / 'valid' / 'images').glob('*.[jp][pn]g')),
        'test': list((dataset_root / 'test' / 'images').glob('*.[jp][pn]g'))
    }

    # Remove empty splits
    splits = {k: v for k, v in splits.items() if v}

    print(f"\nDataset splits found:")
    for split, images in splits.items():
        print(f"  {split}: {len(images)} images")

    # Compute hashes for all images
    print("\nComputing image hashes...")
    hash_to_files = defaultdict(list)

    for split, images in splits.items():
        for img_path in tqdm(images, desc=f"Hashing {split}"):
            img_hash = compute_image_hash(img_path)
            hash_to_files[img_hash].append((split, img_path.name))

    # Find duplicates across splits
    print("\n" + "="*60)
    print("LEAKAGE DETECTION RESULTS")
    print("="*60)

    leakage_found = False
    cross_split_duplicates = []

    for img_hash, occurrences in hash_to_files.items():
        if len(occurrences) > 1:
            # Check if duplicates are in different splits
            splits_involved = set(split for split, _ in occurrences)
            if len(splits_involved) > 1:
                leakage_found = True
                cross_split_duplicates.append((splits_involved, occurrences))

    if leakage_found:
        print(f"\n[WARNING]  DATA LEAKAGE DETECTED! Found {len(cross_split_duplicates)} duplicate images across splits.\n")

        for splits_involved, occurrences in cross_split_duplicates[:10]:  # Show first 10
            print(f"Duplicate found in splits: {', '.join(splits_involved)}")
            for split, filename in occurrences:
                print(f"  - {split}/{filename}")
            print()

        if len(cross_split_duplicates) > 10:
            print(f"... and {len(cross_split_duplicates) - 10} more duplicates")

        print("\n[WARNING]  ACTION REQUIRED: Remove duplicates before training!")
        print("This can significantly affect model performance and validation metrics.")

    else:
        print("\n[OK] No data leakage detected! All splits are clean.")

    # Check for duplicates within splits
    print("\n" + "="*60)
    print("WITHIN-SPLIT DUPLICATE CHECK")
    print("="*60)

    for split, images in splits.items():
        split_hashes = defaultdict(list)
        for img_path in images:
            img_hash = compute_image_hash(img_path)
            split_hashes[img_hash].append(img_path.name)

        duplicates = {h: files for h, files in split_hashes.items() if len(files) > 1}

        if duplicates:
            print(f"\n[WARNING]  {split}: Found {len(duplicates)} duplicate images within split")
            for img_hash, files in list(duplicates.items())[:5]:
                print(f"  {files}")
        else:
            print(f"\n[OK] {split}: No duplicates within split")

    print("\n" + "="*60)


def check_label_quality(dataset_root):
    """Check label quality and consistency."""
    dataset_root = Path(dataset_root)

    print("\n" + "="*60)
    print("LABEL QUALITY CHECK")
    print("="*60)

    splits = ['train', 'val', 'valid', 'test']
    total_issues = 0

    for split in splits:
        label_dir = dataset_root / split / 'labels'
        if not label_dir.exists():
            continue

        label_files = list(label_dir.glob('*.txt'))
        if not label_files:
            continue

        print(f"\nChecking {split} labels...")

        empty_labels = 0
        invalid_format = 0
        out_of_bounds = 0

        for label_file in tqdm(label_files):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                empty_labels += 1
                continue

            for line in lines:
                parts = line.strip().split()

                if len(parts) != 5:
                    invalid_format += 1
                    continue

                try:
                    class_id, x, y, w, h = map(float, parts)

                    # Check bounds
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        out_of_bounds += 1
                        print(f"  Out of bounds: {label_file.name} - x:{x:.3f} y:{y:.3f} w:{w:.3f} h:{h:.3f}")

                    # Check if box is too small
                    if w < 0.01 or h < 0.01:
                        print(f"  Very small box: {label_file.name} - w:{w:.4f} h:{h:.4f}")

                except ValueError:
                    invalid_format += 1

        split_issues = empty_labels + invalid_format + out_of_bounds
        total_issues += split_issues

        print(f"\n{split} results:")
        print(f"  Total labels: {len(label_files)}")
        print(f"  Empty labels: {empty_labels}")
        print(f"  Invalid format: {invalid_format}")
        print(f"  Out of bounds: {out_of_bounds}")

        if split_issues == 0:
            print(f"  [OK] All labels are valid!")
        else:
            print(f"  [WARNING]  Found {split_issues} issues")

    if total_issues > 0:
        print(f"\n[WARNING]  Total issues found: {total_issues}")
        print("Consider cleaning the dataset before training.")
    else:
        print(f"\n[OK] All labels are valid!")


def main():
    parser = argparse.ArgumentParser(description='Check for data leakage and label quality')
    parser.add_argument('--dataset', type=str, default='data/fire_detection',
                       help='Path to dataset root directory')
    parser.add_argument('--skip-leakage', action='store_true',
                       help='Skip leakage detection')
    parser.add_argument('--skip-labels', action='store_true',
                       help='Skip label quality check')

    args = parser.parse_args()

    if not args.skip_leakage:
        check_data_leakage(args.dataset)

    if not args.skip_labels:
        check_label_quality(args.dataset)


if __name__ == "__main__":
    main()
