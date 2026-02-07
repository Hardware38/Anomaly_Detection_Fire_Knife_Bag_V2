"""
Fire Detection Module using YOLOv8
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path


class FireDetector:
    """YOLOv8-based fire detection."""

    def __init__(self, model_path=None, conf_threshold=0.5, iou_threshold=0.45, device='cuda'):
        """
        Initialize fire detector.

        Args:
            model_path: Path to trained model weights (None for pretrained)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'

        if model_path:
            model_path = Path(model_path)
            if model_path.exists():
                print(f"Loading trained model from: {model_path}")
                self.model = YOLO(str(model_path))
            else:
                print(f"Error: Model file not found at: {model_path}")
                print(f"Absolute path checked: {model_path.absolute()}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
        else:
            print("No model path provided. Will use base YOLOv8 for training.")
            self.model = None

    def train(self, data_yaml, epochs=100, img_size=640, batch_size=16, model_size='n',
              lr=0.001, optimizer='Adam', weight_decay=0.0005, patience=50):
        """
        Train fire detection model.

        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Batch size for training
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            lr: Learning rate
            optimizer: Optimizer type
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
        """
        print(f"Training YOLOv8{model_size} fire detection model...")
        print(f"  Learning rate: {lr}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Patience: {patience}")

        # Load base model
        self.model = YOLO(f'yolov8{model_size}.pt')

        # Train with all parameters
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project='results/fire_detection',
            name='yolov8_fire',
            patience=patience,
            save=True,
            plots=True,
            lr0=lr,  # Initial learning rate
            optimizer=optimizer,
            weight_decay=weight_decay,
            # Additional training improvements
            cos_lr=True,  # Cosine LR scheduler
            amp=True,  # Automatic Mixed Precision
            close_mosaic=10  # Disable mosaic augmentation for last 10 epochs
        )

        return results

    def detect(self, image, agument=False):
        """
        Detect fire in image.

        Args:
            image: Input image (numpy array or path)
            agument: Use test-time augmentation for better detection

        Returns:
            detections: List of detection dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        # Run inference with optimized parameters
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            agnostic_nms=True,  # Class-agnostic NMS (daha iyi detection)
            max_det=300,        # Maximum detections per image
            augment=agument     # Test-time augmentation
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])]
                }
                detections.append(detection)

        return detections

    def detect_video(self, video_path, output_path=None, show=True):
        """
        Detect fire in video.

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show: Whether to display video during processing
        """
        cap = cv2.VideoCapture(video_path)

        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        fire_detected_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect fire
            detections = self.detect(frame)

            # Draw detections
            if len(detections) > 0:
                fire_detected_frames += 1
                frame = self.draw_detections(frame, detections)

            if output_path:
                out.write(frame)

            if show:
                cv2.imshow('Fire Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

        print(f"\nVideo processing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Frames with fire detected: {fire_detected_frames}")
        print(f"Fire detection rate: {fire_detected_frames/frame_count*100:.2f}%")

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image."""
        for det in detections:
            bbox = det['bbox'].astype(int)
            conf = det['confidence']
            label = f"{det['class_name']}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (bbox[0], bbox[1] - label_height - 10),
                         (bbox[0] + label_width, bbox[1]), (0, 0, 255), -1)

            # Draw label text
            cv2.putText(image, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def evaluate(self, data_yaml):
        """Evaluate model on test set."""
        if self.model is None:
            raise ValueError("Model not loaded.")

        metrics = self.model.val(data=data_yaml)
        return metrics


if __name__ == "__main__":
    # Example usage
    detector = FireDetector()
    print("Fire Detector initialized")
    print(f"Device: {detector.device}")
