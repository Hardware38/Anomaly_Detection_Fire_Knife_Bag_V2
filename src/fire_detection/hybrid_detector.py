"""
Hybrid Fire Detection System
Combines YOLOv8 deep learning with traditional CV techniques
"""

import cv2
import numpy as np
try:
    from .fire_detector import FireDetector
except ImportError:
    from fire_detector import FireDetector


class HybridFireDetector:
    """
    Hybrid fire detection combining:
    1. YOLOv8 object detection
    2. Color-based fire detection (HSV color space)
    3. Motion detection for dynamic fire behavior
    """

    def __init__(self, yolo_detector, use_color=True, use_motion=True):
        """
        Initialize hybrid detector.

        Args:
            yolo_detector: Trained FireDetector instance
            use_color: Enable color-based detection
            use_motion: Enable motion-based detection
        """
        self.yolo_detector = yolo_detector
        self.use_color = use_color
        self.use_motion = use_motion

        # Fire color ranges in HSV
        # Red-Orange-Yellow colors typical of fire
        self.fire_color_lower1 = np.array([0, 100, 100])
        self.fire_color_upper1 = np.array([10, 255, 255])
        self.fire_color_lower2 = np.array([170, 100, 100])
        self.fire_color_upper2 = np.array([180, 255, 255])

        # Motion detection
        self.prev_frame = None
        self.motion_threshold = 25

    def detect_by_color(self, frame):
        """
        Detect fire-like colors in the frame.

        Args:
            frame: Input image (BGR)

        Returns:
            fire_mask: Binary mask of fire-colored regions
            fire_regions: List of bounding boxes for fire regions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for fire colors (two ranges to cover red wrap-around)
        mask1 = cv2.inRange(hsv, self.fire_color_lower1, self.fire_color_upper1)
        mask2 = cv2.inRange(hsv, self.fire_color_lower2, self.fire_color_upper2)
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fire_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                fire_regions.append({
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'method': 'color'
                })

        return fire_mask, fire_regions

    def detect_motion(self, frame):
        """
        Detect motion in the frame (characteristic of flickering fire).

        Args:
            frame: Input image (BGR)

        Returns:
            motion_mask: Binary mask of motion regions
            has_motion: Boolean indicating significant motion
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return None, False

        # Compute absolute difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        motion_mask = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]

        # Dilate to fill gaps
        motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        # Update previous frame
        self.prev_frame = gray

        # Check if significant motion exists
        has_motion = np.sum(motion_mask) > 1000

        return motion_mask, has_motion

    def detect(self, frame):
        """
        Perform hybrid detection combining all methods.

        Args:
            frame: Input image (BGR)

        Returns:
            results: Dictionary containing all detection results
        """
        results = {
            'yolo_detections': [],
            'color_detections': [],
            'motion_detected': False,
            'fire_detected': False,
            'confidence_score': 0.0
        }

        # 1. YOLO Detection
        if self.yolo_detector.model is not None:
            yolo_dets = self.yolo_detector.detect(frame)
            results['yolo_detections'] = yolo_dets

        # 2. Color-based Detection
        if self.use_color:
            fire_mask, color_regions = self.detect_by_color(frame)
            results['color_detections'] = color_regions
            results['fire_mask'] = fire_mask

        # 3. Motion Detection
        if self.use_motion:
            motion_mask, has_motion = self.detect_motion(frame)
            results['motion_detected'] = has_motion
            results['motion_mask'] = motion_mask

        # 4. Fusion: Combine results from all methods
        results['fire_detected'], results['confidence_score'] = self.fuse_detections(results)

        return results

    def fuse_detections(self, results):
        """
        Fuse detections from multiple methods.

        Scoring system:
        - YOLO detection: High confidence (primary indicator)
        - Color + Motion: Medium confidence
        - Color only: Low confidence

        Args:
            results: Dictionary with detection results

        Returns:
            fire_detected: Boolean
            confidence_score: Float between 0 and 1
        """
        score = 0.0
        weights = {'yolo': 0.6, 'color': 0.3, 'motion': 0.1}

        # YOLO score
        if len(results['yolo_detections']) > 0:
            max_yolo_conf = max([d['confidence'] for d in results['yolo_detections']])
            score += weights['yolo'] * max_yolo_conf

        # Color detection score
        if len(results['color_detections']) > 0:
            color_score = min(len(results['color_detections']) * 0.2, 1.0)
            score += weights['color'] * color_score

        # Motion detection score
        if results['motion_detected']:
            score += weights['motion']

        # Boost score if multiple methods agree
        methods_positive = sum([
            len(results['yolo_detections']) > 0,
            len(results['color_detections']) > 0,
            results['motion_detected']
        ])

        if methods_positive >= 2:
            score = min(score * 1.2, 1.0)

        fire_detected = score > 0.5

        return fire_detected, score

    def visualize(self, frame, results):
        """
        Visualize detection results on frame.

        Args:
            frame: Input image
            results: Detection results dictionary

        Returns:
            vis_frame: Frame with visualizations
        """
        vis_frame = frame.copy()

        # Draw YOLO detections (Red)
        for det in results['yolo_detections']:
            bbox = det['bbox'].astype(int)
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            label = f"YOLO: {det['confidence']:.2f}"
            cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw color detections (Orange)
        for det in results['color_detections']:
            bbox = det['bbox']
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
            label = "Color"
            cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Draw status
        status_text = f"Fire: {'YES' if results['fire_detected'] else 'NO'}"
        conf_text = f"Confidence: {results['confidence_score']:.2f}"
        motion_text = f"Motion: {'YES' if results['motion_detected'] else 'NO'}"

        # Status background
        cv2.rectangle(vis_frame, (10, 10), (300, 100), (0, 0, 0), -1)

        # Status text
        cv2.putText(vis_frame, status_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if results['fire_detected'] else (255, 255, 255), 2)
        cv2.putText(vis_frame, conf_text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, motion_text, (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis_frame


if __name__ == "__main__":
    print("Hybrid Fire Detector Module")
    print("Combines YOLOv8 with color and motion detection")
