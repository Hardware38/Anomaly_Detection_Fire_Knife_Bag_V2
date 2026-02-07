"""
Flask Web Application for Fire Detection
Supports image upload and live camera detection
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from pathlib import Path
import base64
import io
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from fire_detection import FireDetector
from utils.config_loader import ConfigLoader

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
detector = None
config = None
camera = None


def initialize_detector(model_path="results/fire_detection/yolov8_fire8/weights/best.pt"):
    """Initialize fire detection model."""
    global detector, config

    config = ConfigLoader("config.yaml")
    fire_config = config.get_fire_config()

    # Use inference threshold
    conf_threshold = fire_config.get('inference_threshold', fire_config['confidence_threshold'])

    detector = FireDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=fire_config['iou_threshold']
    )

    print(f"Fire detection model loaded successfully!")
    print(f"Confidence threshold: {conf_threshold}")


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and detection."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Detect fire
        detections = detector.detect(image)

        # Draw detections
        output_image = detector.draw_detections(image.copy(), detections)

        # Convert to base64 for web display
        _, buffer = cv2.imencode('.jpg', output_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare results
        fire_detected = len(detections) > 0
        confidence = max([d['confidence'] for d in detections], default=0) if detections else 0

        result = {
            'fire_detected': fire_detected,
            'confidence': float(confidence),
            'num_detections': len(detections),
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': [
                {
                    'confidence': float(d['confidence']),
                    'bbox': [float(x) for x in d['bbox']]  # Convert numpy array to list
                }
                for d in detections
            ]
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500


def generate_camera_frames():
    """Generate camera frames for live detection."""
    global camera

    # Open camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # Detect fire
            detections = detector.detect(frame)

            # Draw detections
            output_frame = detector.draw_detections(frame, detections)

            # Add detection status
            fire_detected = len(detections) > 0
            confidence = max([d['confidence'] for d in detections], default=0) if detections else 0

            # Draw status
            status_text = "FIRE DETECTED!" if fire_detected else "No Fire"
            status_color = (0, 0, 255) if fire_detected else (0, 255, 0)

            cv2.putText(output_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            if fire_detected:
                cv2.putText(output_frame, f"Confidence: {confidence:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        if camera:
            camera.release()


@app.route('/camera_feed')
def camera_feed():
    """Video streaming route."""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera feed."""
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fire Detection Web Application')
    parser.add_argument('--model', type=str,
                       default='results/fire_detection/yolov8_fire8/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run web server')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run web server')

    args = parser.parse_args()

    # Initialize detector
    print("Initializing fire detection model...")
    initialize_detector(args.model)

    # Run app
    print(f"\nStarting web application...")
    print(f"Open your browser and go to: http://{args.host}:{args.port}")
    print(f"\nPress Ctrl+C to stop the server\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
