"""
Real-time fire detection demo
Supports webcam, video file, and image directory
"""

import argparse
import cv2
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from fire_detection import FireDetector, HybridFireDetector
from utils.config_loader import ConfigLoader


class FireDetectionDemo:
    """Demo application for fire detection."""

    def __init__(self, model_path, config_path="config.yaml", use_hybrid=False, use_tta=False):
        """
        Initialize demo.

        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            use_hybrid: Use hybrid detection system
            use_tta: Use Test-Time Augmentation (slower but more accurate)
        """
        self.config = ConfigLoader(config_path)
        fire_config = self.config.get_fire_config()
        self.use_tta = use_tta  # TTA should be optional, not always on

        # Initialize detector
        print("Loading fire detection model...")
        # Use inference_threshold if available, otherwise fall back to confidence_threshold
        conf_threshold = fire_config.get('inference_threshold', fire_config['confidence_threshold'])
        print(f"Using confidence threshold: {conf_threshold}")
        self.yolo_detector = FireDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=fire_config['iou_threshold']
        )

        if use_hybrid:
            print("Initializing hybrid detection system...")
            self.detector = HybridFireDetector(
                yolo_detector=self.yolo_detector,
                use_color=fire_config['hybrid']['use_color_detection'],
                use_motion=fire_config['hybrid']['use_motion_detection']
            )
            self.is_hybrid = True
        else:
            self.detector = self.yolo_detector
            self.is_hybrid = False

        print("Model loaded successfully!")

        # Statistics
        self.frame_count = 0
        self.fire_detections = 0
        self.fps = 0
        self.last_alert_time = 0

        # Alert cooldown (saniye cinsinden - tekrar alarm icin bekleme suresi)
        alert_config = self.config.config.get('alerts', {})
        self.alert_cooldown = alert_config.get('alert_cooldown', 3)
        self.alert_threshold = alert_config.get('fire_alert_threshold', 0.6)

        # Consecutive detection buffer (ard arda tespit - false alarm'i azaltir)
        self.consecutive_detections = 0
        self.consecutive_threshold = 2  # 2 frame ust uste tespit ederse alarm ver (3'ten 2'ye dusuruludu)

        # Temporal smoothing - son N frame'in tespitlerini sakla
        self.detection_history = []
        self.history_size = 5  # Son 5 frame

    def process_frame(self, frame):
        """
        Process a single frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            output_frame: Processed frame with visualizations
            fire_detected: Boolean indicating fire detection
        """
        self.frame_count += 1
        start_time = time.time()

        # Detect
        # Note: TTA is optional and controlled by use_tta flag
        # Using TTA on every frame is VERY slow - only use for difficult cases
        if self.is_hybrid:
            results = self.detector.detect(frame)
            output_frame = self.detector.visualize(frame, results)
            fire_detected = results['fire_detected']
            confidence = results['confidence_score']
        else:
            # Only use TTA if explicitly enabled (slower but more accurate)
            detections = self.detector.detect(frame, agument=self.use_tta)
            output_frame = self.detector.draw_detections(frame, detections)
            fire_detected = len(detections) > 0
            confidence = max([d['confidence'] for d in detections], default=0)

        # Temporal smoothing - son frame'lerin ortalamasini al
        self.detection_history.append((fire_detected, confidence))
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)

        # Son N frame'in cogunlugunda ates varsa, gercekten ates var
        recent_detections = sum(1 for det, _ in self.detection_history if det)
        smoothed_fire_detected = recent_detections >= (len(self.detection_history) // 2)

        # CRITICAL FIX: Only count UNIQUE detections
        # If this is a NEW detection (previous frame didn't have fire), count it
        # This prevents counting the same fire 100 times (once per frame)
        if smoothed_fire_detected:
            # Only increment if this is a new detection event
            if self.consecutive_detections == 0:
                self.fire_detections += 1
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        # Calculate FPS
        elapsed = time.time() - start_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0

        # Draw statistics (smoothed result kullan)
        self.draw_statistics(output_frame, smoothed_fire_detected, confidence)

        # Alert (ard arda tespit kontrolu ile)
        # Hem confidence yuksek olmali, hem de ard arda tespit edilmeli
        if (smoothed_fire_detected and
            confidence > self.alert_threshold and
            self.consecutive_detections >= self.consecutive_threshold):
            self.trigger_alert(confidence)

        return output_frame, smoothed_fire_detected

    def draw_statistics(self, frame, fire_detected, confidence):
        """Draw statistics overlay on frame."""
        h, w = frame.shape[:2]

        # Statistics panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 300, 0), (w, panel_height), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Statistics text
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Detections: {self.fire_detections}",
            f"Fire: {'YES' if fire_detected else 'NO'}",
        ]

        y_offset = 25
        for stat in stats:
            color = (0, 255, 0) if fire_detected and "Fire:" in stat else (255, 255, 255)
            cv2.putText(frame, stat, (w - 290, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        # Detection rate
        detection_rate = (self.fire_detections / self.frame_count * 100) if self.frame_count > 0 else 0
        cv2.putText(frame, f"Rate: {detection_rate:.1f}%", (w - 290, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def trigger_alert(self, confidence):
        """Trigger fire alert."""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            print(f"\n{'='*60}")
            print(f"🔥 FIRE ALERT! Confidence: {confidence:.2f}")
            print(f"{'='*60}\n")
            self.last_alert_time = current_time

    def run_webcam(self, camera_id=0):
        """
        Run detection on webcam.

        Args:
            camera_id: Camera device ID (default: 0)
        """
        print(f"\nStarting webcam detection (Camera {camera_id})...")
        print("Press 'q' to quit, 's' to save screenshot")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Process frame
            output_frame, fire_detected = self.process_frame(frame)

            # Display
            cv2.imshow('Fire Detection - Press Q to quit', output_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"results/screenshot_{self.frame_count}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"Saved screenshot: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        self.print_summary()

    def run_video(self, video_path, output_path=None, display=True):
        """
        Run detection on video file.

        Args:
            video_path: Path to video file
            output_path: Path to save output video (optional)
            display: Whether to display video
        """
        print(f"\nProcessing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")

        # Process video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            output_frame, fire_detected = self.process_frame(frame)

            # Save
            if output_path:
                out.write(output_frame)

            # Display
            if display:
                cv2.imshow('Fire Detection - Press Q to quit', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames * 100)
                print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})", end='\r')

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        self.print_summary()

    def run_image(self, image_path, output_path=None, display=True):
        """
        Run detection on single image.

        Args:
            image_path: Path to image
            output_path: Path to save output (optional)
            display: Whether to display image
        """
        print(f"\nProcessing image: {image_path}")

        # Read image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return

        # Process
        output_frame, fire_detected = self.process_frame(frame)

        # Save
        if output_path:
            cv2.imwrite(str(output_path), output_frame)
            print(f"Saved output to: {output_path}")

        # Display
        if display:
            cv2.imshow('Fire Detection - Press any key to close', output_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.print_summary()

    def print_summary(self):
        """Print detection summary."""
        print(f"\n{'='*60}")
        print("DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Frames with fire: {self.fire_detections}")
        if self.frame_count > 0:
            print(f"Detection rate: {self.fire_detections/self.frame_count*100:.2f}%")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Fire Detection Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--source', type=str, default='0',
                       help='Source: 0 for webcam, path to video/image file')
    parser.add_argument('--output', type=str,
                       help='Path to save output video/image')
    parser.add_argument('--hybrid', action='store_true',
                       help='Use hybrid detection system')
    parser.add_argument('--tta', action='store_true',
                       help='Use Test-Time Augmentation (slower but more accurate)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output')

    args = parser.parse_args()

    # Initialize demo
    demo = FireDetectionDemo(
        model_path=args.model,
        config_path=args.config,
        use_hybrid=args.hybrid,
        use_tta=args.tta
    )

    # Determine source type
    if args.source == '0' or args.source.isdigit():
        # Webcam
        demo.run_webcam(camera_id=int(args.source))
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: Source not found: {args.source}")
            return

        # Check if video or image
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        if source_path.suffix.lower() in video_extensions:
            demo.run_video(source_path, args.output, not args.no_display)
        elif source_path.suffix.lower() in image_extensions:
            demo.run_image(source_path, args.output, not args.no_display)
        else:
            print(f"Error: Unsupported file format: {source_path.suffix}")


if __name__ == "__main__":
    main()
