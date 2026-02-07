"""
Batch image testing script
Test multiple images at once and save results
"""

import argparse
from pathlib import Path
import subprocess
from tqdm import tqdm


def test_images(model_path, image_dir, output_dir, use_tta=False):
    """
    Test multiple images with fire detection model.

    Args:
        model_path: Path to trained model
        image_dir: Directory containing images to test
        output_dir: Directory to save results
        use_tta: Use Test-Time Augmentation
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(ext)))
        image_files.extend(list(image_dir.glob(ext.upper())))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Testing with model: {model_path}")
    print(f"Results will be saved to: {output_dir}")
    print("="*60)

    # Test each image
    results = []
    for img_path in tqdm(image_files, desc="Testing images"):
        output_path = output_dir / f"result_{img_path.name}"

        # Build command
        cmd = [
            "python", "src/fire_detection/demo.py",
            "--model", str(model_path),
            "--source", str(img_path),
            "--output", str(output_path),
            "--no-display"
        ]

        if use_tta:
            cmd.append("--tta")

        # Run detection
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Check if fire was detected
            if "Frames with fire: 1" in result.stdout:
                fire_detected = True
                # Extract confidence if available
                confidence = "N/A"
            else:
                fire_detected = False
                confidence = "N/A"

            results.append({
                'image': img_path.name,
                'fire_detected': fire_detected,
                'confidence': confidence,
                'output': output_path
            })

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            results.append({
                'image': img_path.name,
                'fire_detected': 'ERROR',
                'confidence': 'N/A',
                'output': None
            })

    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    fire_count = sum(1 for r in results if r['fire_detected'] is True)
    no_fire_count = sum(1 for r in results if r['fire_detected'] is False)
    error_count = sum(1 for r in results if r['fire_detected'] == 'ERROR')

    print(f"\nTotal images tested: {len(results)}")
    print(f"Fire detected: {fire_count}")
    print(f"No fire detected: {no_fire_count}")
    print(f"Errors: {error_count}")

    print("\nDetailed results:")
    for r in results:
        status = "FIRE" if r['fire_detected'] is True else "NO FIRE" if r['fire_detected'] is False else "ERROR"
        print(f"  {r['image']}: {status}")

    print(f"\nAll results saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test multiple images with fire detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images to test')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Directory to save results')
    parser.add_argument('--tta', action='store_true',
                       help='Use Test-Time Augmentation')

    args = parser.parse_args()

    test_images(args.model, args.images, args.output, args.tta)


if __name__ == "__main__":
    main()
