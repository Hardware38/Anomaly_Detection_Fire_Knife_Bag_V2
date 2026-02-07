"""
Testing and evaluation script for fire detection
"""

import argparse
import cv2
import torch
from pathlib import Path
import numpy as np
from fire_detector import FireDetector
from hybrid_detector import HybridFireDetector
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import ConfigLoader


def calculate_metrics(predictions, ground_truth):
    """
    Calculate detection metrics.

    Args:
        predictions: List of predicted boxes
        ground_truth: List of ground truth boxes

    Returns:
        metrics: Dictionary of metric values
    """
    # Simplified metrics calculation
    # In practice, use proper IoU-based matching
    tp = len([p for p in predictions if p['confidence'] > 0.5])
    fp = len([p for p in predictions if p['confidence'] <= 0.5])
    fn = max(0, len(ground_truth) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def test_on_image(detector, image_path, output_path=None, visualize=True):
    """
    Test detector on a single image.

    Args:
        detector: FireDetector or HybridFireDetector instance
        image_path: Path to test image
        output_path: Path to save output image
        visualize: Whether to display the result
    """
    print(f"\nTesting on: {image_path}")

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Detect
    if isinstance(detector, HybridFireDetector):
        results = detector.detect(image)
        output_image = detector.visualize(image, results)
        print(f"Fire detected: {results['fire_detected']}")
        print(f"Confidence: {results['confidence_score']:.3f}")
        print(f"YOLO detections: {len(results['yolo_detections'])}")
        print(f"Color detections: {len(results['color_detections'])}")
        print(f"Motion detected: {results['motion_detected']}")
    else:
        detections = detector.detect(image)
        output_image = detector.draw_detections(image, detections)
        print(f"Detections: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")

    # Save output
    if output_path:
        cv2.imwrite(str(output_path), output_image)
        print(f"Saved output to: {output_path}")

    # Display
    if visualize:
        cv2.imshow('Fire Detection', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_on_video(detector, video_path, output_path=None, show=True):
    """
    Test detector on video.

    Args:
        detector: FireDetector or HybridFireDetector instance
        video_path: Path to test video
        output_path: Path to save output video
        show: Whether to display video during processing
    """
    print(f"\nTesting on video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    fire_frames = 0
    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect
        if isinstance(detector, HybridFireDetector):
            results = detector.detect(frame)
            output_frame = detector.visualize(frame, results)
            if results['fire_detected']:
                fire_frames += 1
                total_detections += len(results['yolo_detections'])
        else:
            detections = detector.detect(frame)
            output_frame = detector.draw_detections(frame, detections)
            if len(detections) > 0:
                fire_frames += 1
                total_detections += len(detections)

        # Save frame
        if output_path:
            out.write(output_frame)

        # Display
        if show:
            cv2.imshow('Fire Detection', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...", end='\r')

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    print(f"\n\nVideo testing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Frames with fire: {fire_frames} ({fire_frames/frame_count*100:.1f}%)")
    print(f"Total detections: {total_detections}")
    print(f"Avg detections per frame: {total_detections/frame_count:.2f}")


def test_on_dataset(detector, data_yaml_path):
    """
    Evaluate detector on test dataset.

    Args:
        detector: FireDetector instance
        data_yaml_path: Path to data.yaml
    """
    print("\nEvaluating on test dataset...")
    metrics = detector.evaluate(data_yaml_path)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    print("="*60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test fire detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'dataset'],
                       default='image', help='Test mode')
    parser.add_argument('--input', type=str, help='Path to input image/video')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--data-yaml', type=str, help='Path to data.yaml for dataset evaluation')
    parser.add_argument('--hybrid', action='store_true',
                       help='Use hybrid detection system')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display output')

    args = parser.parse_args()

    # Load configuration
    config = ConfigLoader(args.config)
    fire_config = config.get_fire_config()

    # Initialize detector
    print("Loading model...")
    yolo_detector = FireDetector(
        model_path=args.model,
        conf_threshold=fire_config['confidence_threshold'],
        iou_threshold=fire_config['iou_threshold']
    )

    if args.hybrid:
        print("Using hybrid detection system")
        detector = HybridFireDetector(
            yolo_detector=yolo_detector,
            use_color=fire_config['hybrid']['use_color_detection'],
            use_motion=fire_config['hybrid']['use_motion_detection']
        )
    else:
        detector = yolo_detector

    # Run tests
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        test_on_image(detector, args.input, args.output, not args.no_show)

    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        test_on_video(detector, args.input, args.output, not args.no_show)

    elif args.mode == 'dataset':
        if not args.data_yaml:
            print("Error: --data-yaml required for dataset mode")
            return
        test_on_dataset(yolo_detector, args.data_yaml)


if __name__ == "__main__":
    main()
