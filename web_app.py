"""
Flask Web Application for Multi-Model Detection
Supports fire detection, knife detection, bag detection, and train detection
Supports image upload, video upload and live camera detection
"""

from flask import Flask, render_template, request, jsonify, Response
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
import cv2
import numpy as np
from pathlib import Path
import base64
import io
import tempfile
import os
import threading
import uuid
import time
from collections import deque
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from fire_detection import FireDetector, CascadeFireDetector, BagCascadeDetector
from fire_detection.cascade_detector import extract_crop
from fire_detection.decision_fusion import DecisionFusion
from knife_detection import KnifeCascadeDetector
from tracking import ObjectTracker, AbandonedBagMonitor
from utils.config_loader import ConfigLoader

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size (videos)
if CORS is not None:
    CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
detectors = {}  # {'fire': FireDetector, 'knife': FireDetector, ...}
trackers = {}   # {'fire': ObjectTracker, 'knife': ObjectTracker, ...}
cascade = None  # CascadeFireDetector (if classifier weights exist)
fusion = None   # DecisionFusion (brightness + motion yardımcı karar)
bag_cascade = None
knife_cascade = None
bag_monitor = None
bag_abandonment_config = {}
bag_detection_config = {}
knife_detection_config = {}
fire_postfilter_config = {}
camera_runtime_config = {}
stream_validation_config = {}
config = None
camera = None
camera_models = []  # Active models for camera feed
video_jobs = {}  # {job_id: {'status': ..., 'progress': ..., 'total_frames': ..., 'fire_frames': ..., 'path': ...}}
camera_validator = None
phone_validator = None

# Tracking configuration (temporal smoothing)
TRACKING_MAX_AGE = 15   # 15 frames missed → DELETE (ateş titreşse bile takip devam eder)
TRACKING_MIN_HITS = 3   # 3 consecutive frames → CONFIRMED (alarm enabled)

# Synthetic fire boxes must also pass the classifier.
SYNTHETIC_VALIDATION_THRESHOLD = 0.55

# Detection colors per model
MODEL_COLORS = {
    'fire': (0, 0, 255),      # Red
    'knife': (0, 165, 255),    # Orange
    'bag': (255, 165, 0),      # Blue-ish orange
    'luggage': (255, 100, 0),  # Darker blue
    'person': (0, 255, 0),     # Green
    'train': (168, 85, 247),   # Purple
}

MODEL_LABELS = {
    'fire': 'YANGIN',
    'knife': 'BICAK',
    'bag': 'CANTA',
    'luggage': 'BAVUL',
    'person': 'KISI',
    'train': 'TREN',
}

# bag_best.pt has multiple classes - map them to separate display keys
DEFAULT_BAG_MODEL_CLASSES = ('bag', 'luggage', 'person')
BAG_MODEL_CLASSES = set(DEFAULT_BAG_MODEL_CLASSES)
TRAIN_MODEL_CLASSES = {'train'}
ABANDONED_BAG_MODELS = {'bag', 'luggage'}
MODEL_CLASS_GROUPS = {
    'bag': BAG_MODEL_CLASSES,
    'train': TRAIN_MODEL_CLASSES,
}


def _normalize_class_names(values, fallback):
    """Normalize config class lists into a lowercase set."""
    normalized = {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }
    return normalized or set(fallback)


def _ordered_bag_classes(values):
    """Keep bag/luggage/person order stable for UI and classifier heads."""
    preferred = [name for name in DEFAULT_BAG_MODEL_CLASSES if name in values]
    extras = sorted(name for name in values if name not in preferred)
    return preferred + extras


def _filter_model_detections(model_name, detections, image_shape=None):
    """Keep only the classes that should be exposed for a multi-class model."""
    allowed_classes = MODEL_CLASS_GROUPS.get(model_name)
    if not allowed_classes:
        return detections
    detections = [
        det for det in detections
        if str(det.get('class_name', '')).lower() in allowed_classes
    ]
    if model_name == 'bag':
        detections = _filter_bag_detections(detections, image_shape)
    return detections


def _filter_bag_detections(detections, image_shape):
    """Apply minimal post-filtering for bestSON bag/person/luggage detections."""
    if image_shape is None:
        return detections

    frame_h, frame_w = image_shape[:2]
    frame_area = max(float(frame_h * frame_w), 1.0)

    cfg = {
        # Reference desktop web_app.py works better with a confidence-first bag path.
        'bag_confidence_threshold': 0.45,
        'person_confidence_threshold': 0.45,
        'min_width': 1,
        'min_height': 1,
        'min_area_ratio': 0.0,
        'max_area_ratio': 1.0,
        'min_aspect_ratio': 0.05,
        'max_aspect_ratio': 20.00,
        'edge_margin_ratio': 0.0,
        'edge_confidence_threshold': 0.45,
        'hide_tentative_tracks': False,
    }
    cfg.update(bag_detection_config or {})

    filtered = []
    edge_margin_x = frame_w * float(cfg['edge_margin_ratio'])
    edge_margin_y = frame_h * float(cfg['edge_margin_ratio'])

    for det in detections:
        cls_name = str(det.get('class_name', '')).lower()
        conf = float(det.get('confidence', 0.0))
        bbox = np.asarray(det.get('bbox', [0, 0, 0, 0]), dtype=float)
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        class_threshold = (
            float(cfg['person_confidence_threshold'])
            if cls_name == 'person'
            else float(cfg['bag_confidence_threshold'])
        )
        if conf < class_threshold:
            continue
        if width < float(cfg['min_width']) or height < float(cfg['min_height']):
            continue

        area_ratio = (width * height) / frame_area
        if area_ratio < float(cfg['min_area_ratio']) or area_ratio > float(cfg['max_area_ratio']):
            continue

        aspect_ratio = width / max(height, 1.0)
        if aspect_ratio < float(cfg['min_aspect_ratio']) or aspect_ratio > float(cfg['max_aspect_ratio']):
            continue

        touches_edge = (
            x1 <= edge_margin_x or y1 <= edge_margin_y or
            x2 >= (frame_w - edge_margin_x) or y2 >= (frame_h - edge_margin_y)
        )
        if touches_edge and conf < float(cfg['edge_confidence_threshold']):
            continue

        filtered.append(det)

    return filtered


def _resolve_model_key(model_name, class_name):
    """Map multi-class detector outputs to the UI/result key."""
    class_name = str(class_name or '').lower()
    allowed_classes = MODEL_CLASS_GROUPS.get(model_name)
    if allowed_classes and class_name in allowed_classes:
        return class_name
    return model_name


def _display_keys(active_models):
    """Expand model keys used in UI summaries."""
    keys = set(active_models)
    for model_name in active_models:
        keys.update(MODEL_CLASS_GROUPS.get(model_name, set()))
    return keys


def _apply_bag_monitor(monitor, tracked_objects, now):
    """Run abandoned bag logic for bag/person tracks."""
    if monitor is None:
        return {'warning_count': 0, 'abandoned_count': 0, 'bag_count': 0, 'person_count': 0}
    if not any(obj.get('model') in BAG_MODEL_CLASSES for obj in tracked_objects):
        return {'warning_count': 0, 'abandoned_count': 0, 'bag_count': 0, 'person_count': 0}
    return monitor.update(tracked_objects, now)


def _is_alarm_object(obj):
    """Return True when an object should raise an alarm."""
    model = obj.get('model')
    if model == 'person':
        return False
    if model in ABANDONED_BAG_MODELS:
        return bool(obj.get('bag_alert_warning') or obj.get('bag_alert_abandoned'))
    return True


def _alarm_text_for_objects(model_name, objects):
    """Return overlay text for an alarm bucket."""
    if model_name in ABANDONED_BAG_MODELS:
        if any(obj.get('bag_alert_abandoned') for obj in objects):
            return 'TERK EDILMIS CANTA!'
        if any(obj.get('bag_alert_warning') for obj in objects):
            return 'SUPHELI CANTA!'
        return None
    label = MODEL_LABELS.get(model_name, model_name.upper())
    return f"{label} TESPIT EDILDI!"


def _resolve_bag_model_path(requested_path):
    """Prefer the unified bestSON bag/person/luggage model when available."""
    requested = Path(requested_path)
    preferred = Path('bestSON.pt')
    if preferred.exists():
        return str(preferred)
    return str(requested)


def _compute_bbox_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _compute_bbox_cover(box_a, box_b):
    """Return how much of box_a is covered by box_b."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    return inter / area_a if area_a > 0 else 0.0


def _merge_nested_dicts(base, override):
    """Recursively merge override into base without mutating inputs."""
    result = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_nested_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _resize_for_inference(image, max_width=None):
    """Downscale large frames for faster live inference and return the scale factor."""
    if not max_width:
        return image, 1.0

    height, width = image.shape[:2]
    max_width = int(max_width)
    if max_width <= 0 or width <= max_width:
        return image, 1.0

    scale = max_width / max(float(width), 1.0)
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _scale_detection_bboxes(detections, scale):
    """Map bboxes from the resized inference frame back to original coordinates."""
    if not detections or scale == 1.0:
        return detections

    inv_scale = 1.0 / scale
    scaled = []
    for det in detections:
        det_copy = dict(det)
        det_copy['bbox'] = np.asarray(det['bbox'], dtype=float) * inv_scale
        scaled.append(det_copy)
    return scaled


def _open_live_camera(index=0, backend_name='dshow'):
    """Open webcam with a Windows-friendly backend fallback order."""
    backend_map = {
        'dshow': getattr(cv2, 'CAP_DSHOW', cv2.CAP_ANY),
        'msmf': getattr(cv2, 'CAP_MSMF', cv2.CAP_ANY),
        'any': cv2.CAP_ANY,
    }
    preferred = str(backend_name or 'dshow').strip().lower()
    order = [preferred]
    for name in ('dshow', 'msmf', 'any'):
        if name not in order:
            order.append(name)

    for name in order:
        cap = cv2.VideoCapture(int(index), backend_map.get(name, cv2.CAP_ANY))
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    return None


def _fire_quality_score(det):
    """Prefer classifier probability when available, otherwise fall back to confidence."""
    cls_prob = det.get('classifier_prob')
    if cls_prob is not None:
        return float(cls_prob)
    return float(det.get('confidence', 0.0))


def _analyze_fire_crop_appearance(image, bbox):
    """Measure whether a fire crop looks like a smooth warm object or bright glare."""
    x1, y1, x2, y2 = np.asarray(bbox, dtype=int)
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, image.shape[1]))
    y2 = max(y1 + 1, min(y2, image.shape[0]))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    warm_mask = (
        (h_ch <= 32) &
        (s_ch >= 70) &
        (v_ch >= 140)
    )
    yellow_mask = (
        (h_ch >= 18) & (h_ch <= 42) &
        (s_ch >= 90) &
        (v_ch >= 170)
    )
    white_hot_mask = (
        (s_ch <= 40) &
        (v_ch >= 210)
    )
    bright_mask = v_ch >= 220
    edges = cv2.Canny(gray, 80, 160)

    return {
        'warm_ratio': float(np.count_nonzero(warm_mask)) / max(float(warm_mask.size), 1.0),
        'yellow_ratio': float(np.count_nonzero(yellow_mask)) / max(float(yellow_mask.size), 1.0),
        'white_ratio': float(np.count_nonzero(white_hot_mask)) / max(float(white_hot_mask.size), 1.0),
        'bright_ratio': float(np.count_nonzero(bright_mask)) / max(float(bright_mask.size), 1.0),
        'edge_ratio': float(np.count_nonzero(edges)) / max(float(edges.size), 1.0),
        'hue_std': float(h_ch.astype(np.float32).std()),
        'mean_saturation': float(s_ch.astype(np.float32).mean()),
        'mean_value': float(v_ch.astype(np.float32).mean()),
    }


def _is_uniform_warm_object(metrics, cfg):
    """Reject smooth yellow objects that often fool the fire classifier."""
    if not metrics:
        return False
    return (
        metrics['yellow_ratio'] >= float(cfg.get('min_yellow_ratio', 0.55)) and
        metrics['edge_ratio'] <= float(cfg.get('max_edge_ratio', 0.09)) and
        metrics['hue_std'] <= float(cfg.get('max_hue_std', 16.0))
    )


def _is_bright_glare_object(metrics, cfg):
    """Reject sun glare and bright white highlights that decision fusion can mistake for fire."""
    if not metrics:
        return False
    return (
        metrics['white_ratio'] >= float(cfg.get('min_white_ratio', 0.42)) and
        metrics['bright_ratio'] >= float(cfg.get('min_bright_ratio', 0.35)) and
        metrics['warm_ratio'] <= float(cfg.get('max_warm_ratio', 0.06)) and
        metrics['mean_saturation'] <= float(cfg.get('max_mean_saturation', 90.0))
    )


def _filter_fire_false_positives(image, detections, override_cfg=None):
    """Suppress common false positives such as yellow bags and coats."""
    cfg = {
        'enabled': True,
        'minimum_fire_quality_score': None,
        'minimum_box_width': 0,
        'minimum_box_height': 0,
        'minimum_box_area_ratio': 0.0,
        'reject_synthetic': False,
        'small_box_guard': {
            'enabled': False,
            'max_width': 42,
            'max_height': 42,
            'max_area_ratio': 0.0010,
            'minimum_quality_score': 0.82,
            'reject_synthetic': True,
        },
        'uniform_warm_object': {
            'enabled': True,
            'min_yellow_ratio': 0.55,
            'max_edge_ratio': 0.09,
            'max_hue_std': 16.0,
            'classifier_override': 0.85,
        },
        'bright_glare_object': {
            'enabled': True,
            'synthetic_only': True,
            'min_white_ratio': 0.42,
            'min_bright_ratio': 0.35,
            'max_warm_ratio': 0.06,
            'max_mean_saturation': 90.0,
            'classifier_override': 0.96,
        },
        'auxiliary_object_veto': {
            'enabled': True,
            'target_classes': ['bag', 'luggage', 'person'],
            'classifier_override': 0.85,
            'min_iou': 0.20,
            'min_cover': 0.55,
            'synthetic_full_cover_veto': {
                'enabled': True,
                'target_classes': ['bag', 'luggage', 'person'],
                'min_cover': 0.95,
                'center_inside': True,
                'require_glare_metrics': False,
                'min_white_ratio': 0.08,
                'max_warm_ratio': 0.08,
                'max_mean_saturation': 135.0,
                'classifier_override': 0.995,
            },
        },
    }
    cfg = _merge_nested_dicts(cfg, fire_postfilter_config or {})
    cfg = _merge_nested_dicts(cfg, override_cfg or {})
    if not bool(cfg.get('enabled', True)):
        return detections

    uniform_cfg = dict(cfg.get('uniform_warm_object', {}) or {})
    glare_cfg = dict(cfg.get('bright_glare_object', {}) or {})
    aux_cfg = dict(cfg.get('auxiliary_object_veto', {}) or {})
    small_cfg = dict(cfg.get('small_box_guard', {}) or {})
    synthetic_cover_cfg = dict(aux_cfg.get('synthetic_full_cover_veto', {}) or {})
    filtered = []
    aux_candidates = []
    synthetic_cover_candidates = []
    frame_area = max(float(image.shape[0] * image.shape[1]), 1.0)

    for det in detections:
        if det.get('class_name') != 'fire':
            filtered.append(det)
            continue

        bbox = np.asarray(det['bbox'], dtype=float)
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        area_ratio = (width * height) / frame_area
        if width < float(cfg.get('minimum_box_width', 0)):
            continue
        if height < float(cfg.get('minimum_box_height', 0)):
            continue
        if area_ratio < float(cfg.get('minimum_box_area_ratio', 0.0)):
            continue

        quality_score = _fire_quality_score(det)
        if bool(cfg.get('reject_synthetic', False)) and det.get('fusion_synthetic'):
            continue
        is_small_box = (
            width <= float(small_cfg.get('max_width', 42)) or
            height <= float(small_cfg.get('max_height', 42)) or
            area_ratio <= float(small_cfg.get('max_area_ratio', 0.0010))
        )
        if bool(small_cfg.get('enabled', False)) and is_small_box:
            if bool(small_cfg.get('reject_synthetic', True)) and det.get('fusion_synthetic'):
                continue
            if quality_score < float(small_cfg.get('minimum_quality_score', 0.82)):
                continue
        min_quality_score = cfg.get('minimum_fire_quality_score')
        if min_quality_score is not None and quality_score < float(min_quality_score):
            continue
        metrics = _analyze_fire_crop_appearance(image, det['bbox'])
        if (
            bool(glare_cfg.get('enabled', True)) and
            _is_bright_glare_object(metrics, glare_cfg) and
            (
                not bool(glare_cfg.get('synthetic_only', True)) or
                det.get('fusion_synthetic')
            ) and
            quality_score < float(glare_cfg.get('classifier_override', 0.96))
        ):
            continue
        if (
            bool(uniform_cfg.get('enabled', True)) and
            _is_uniform_warm_object(metrics, uniform_cfg) and
            quality_score < float(uniform_cfg.get('classifier_override', 0.85))
        ):
            continue

        det_copy = dict(det)
        det_copy['fire_appearance_metrics'] = metrics
        filtered.append(det_copy)
        if bool(synthetic_cover_cfg.get('enabled', True)) and det.get('fusion_synthetic'):
            synthetic_cover_candidates.append(det_copy)

        if (
            bool(aux_cfg.get('enabled', True)) and
            quality_score < float(aux_cfg.get('classifier_override', 0.85))
        ):
            aux_candidates.append(det_copy)

    if (not aux_candidates and not synthetic_cover_candidates) or 'bag' not in detectors:
        return filtered

    target_classes = _normalize_class_names(
        aux_cfg.get('target_classes'),
        DEFAULT_BAG_MODEL_CLASSES,
    )
    auxiliary_detections = [
        det for det in _bag_detect_with_classifier(image)
        if str(det.get('class_name', '')).lower() in target_classes
    ]
    if not auxiliary_detections:
        return filtered

    keep = []
    min_iou = float(aux_cfg.get('min_iou', 0.20))
    min_cover = float(aux_cfg.get('min_cover', 0.55))
    synthetic_cover_targets = _normalize_class_names(
        synthetic_cover_cfg.get('target_classes'),
        target_classes,
    )
    for det in filtered:
        if det.get('class_name') != 'fire':
            keep.append(det)
            continue
        quality_score = _fire_quality_score(det)
        metrics = det.get('fire_appearance_metrics') or {}
        if quality_score >= float(aux_cfg.get('classifier_override', 0.85)):
            allow_regular_veto = False
        else:
            allow_regular_veto = True

        fire_bbox = np.asarray(det['bbox'], dtype=float)
        vetoed = False
        for aux_det in auxiliary_detections:
            aux_bbox = np.asarray(aux_det['bbox'], dtype=float)
            aux_class = str(aux_det.get('class_name', '')).lower()
            cover = _compute_bbox_cover(fire_bbox, aux_bbox)
            center_x = (float(fire_bbox[0]) + float(fire_bbox[2])) * 0.5
            center_y = (float(fire_bbox[1]) + float(fire_bbox[3])) * 0.5
            center_inside = (
                center_x >= float(aux_bbox[0]) and center_x <= float(aux_bbox[2]) and
                center_y >= float(aux_bbox[1]) and center_y <= float(aux_bbox[3])
            )
            require_glare_metrics = bool(synthetic_cover_cfg.get('require_glare_metrics', False))
            glare_like = (
                metrics.get('white_ratio', 0.0) >= float(synthetic_cover_cfg.get('min_white_ratio', 0.08)) and
                metrics.get('warm_ratio', 1.0) <= float(synthetic_cover_cfg.get('max_warm_ratio', 0.08)) and
                metrics.get('mean_saturation', 255.0) <= float(synthetic_cover_cfg.get('max_mean_saturation', 135.0))
            )
            if (
                bool(synthetic_cover_cfg.get('enabled', True)) and
                det.get('fusion_synthetic') and
                aux_class in synthetic_cover_targets and
                (
                    cover >= float(synthetic_cover_cfg.get('min_cover', 0.95)) or
                    (
                        bool(synthetic_cover_cfg.get('center_inside', True)) and
                        center_inside
                    )
                ) and
                (
                    not require_glare_metrics or
                    glare_like
                ) and
                quality_score < float(synthetic_cover_cfg.get('classifier_override', 0.995))
            ):
                vetoed = True
                break
            if not allow_regular_veto:
                continue
            if (
                _compute_bbox_iou(fire_bbox, aux_bbox) >= min_iou or
                cover >= min_cover
            ):
                vetoed = True
                break
        if not vetoed:
            keep.append(det)

    return keep


class StreamDetectionValidator:
    """Per-track temporal validation to reduce live-camera false positives."""

    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = bool(cfg.get('enabled', True))
        self.history_size = int(cfg.get('history_size', 12))
        self.motion_diff_threshold = int(cfg.get('motion_diff_threshold', 18))
        self.global_motion_threshold = float(cfg.get('global_motion_threshold', 0.18))
        self.max_missed_frames = int(cfg.get('max_missed_frames', 2))
        self.model_rules = {
            'default': {
                'min_seen_frames': 3,
                'min_duration': 0.25,
                'min_avg_confidence': 0.40,
                'min_area_ratio': 0.0,
                'min_width': 1,
                'min_height': 1,
                'alarm_min_seen_frames': 3,
            },
            'fire': {
                'min_seen_frames': 4,
                'min_duration': 0.45,
                'min_avg_confidence': 0.40,
                'min_area_ratio': 0.0002,
                'min_width': 12,
                'min_height': 12,
                'min_motion_ratio': 0.03,
                'min_motion_frames': 3,
                'alarm_min_motion_frames': 4,
                'local_motion_vs_global': 0.55,
                'allow_fusion_evidence': True,
            },
            'knife': {
                'min_seen_frames': 4,
                'min_duration': 0.40,
                'min_avg_confidence': 0.48,
                'min_area_ratio': 0.00008,
                'min_width': 14,
                'min_height': 14,
                'alarm_min_seen_frames': 5,
            },
            'bag': {
                'min_seen_frames': 3,
                'min_duration': 0.30,
                'min_avg_confidence': 0.42,
                'min_area_ratio': 0.00025,
                'min_width': 18,
                'min_height': 18,
                'alarm_min_seen_frames': 3,
            },
            'luggage': {
                'min_seen_frames': 3,
                'min_duration': 0.30,
                'min_avg_confidence': 0.42,
                'min_area_ratio': 0.00030,
                'min_width': 20,
                'min_height': 20,
                'alarm_min_seen_frames': 3,
            },
            'person': {
                'min_seen_frames': 2,
                'min_duration': 0.15,
                'min_avg_confidence': 0.30,
                'min_area_ratio': 0.0005,
                'min_width': 18,
                'min_height': 24,
                'alarm_min_seen_frames': 2,
            },
            'train': {
                'min_seen_frames': 3,
                'min_duration': 0.30,
                'min_avg_confidence': 0.45,
                'min_area_ratio': 0.0010,
                'min_width': 30,
                'min_height': 24,
                'alarm_min_seen_frames': 3,
            },
        }
        self._merge_model_rules(cfg.get('models', {}))
        self.prev_gray = None
        self.track_states = {}

    def _merge_model_rules(self, config_models):
        for model_name, values in (config_models or {}).items():
            merged = dict(self.model_rules.get('default', {}))
            merged.update(self.model_rules.get(model_name, {}))
            merged.update(values or {})
            self.model_rules[model_name] = merged

    def reset(self):
        self.prev_gray = None
        self.track_states.clear()

    def _rules_for(self, model_name):
        rules = dict(self.model_rules.get('default', {}))
        rules.update(self.model_rules.get(model_name, {}))
        return rules

    def _track_key(self, obj):
        return (obj.get('model'), obj.get('track_id'))

    def _prepare_motion_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return gray, None, 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        _, motion_mask = cv2.threshold(diff, self.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, None, iterations=1)
        self.prev_gray = gray
        global_motion = float(np.count_nonzero(motion_mask)) / max(float(motion_mask.size), 1.0)
        return gray, motion_mask, global_motion

    def _local_motion_ratio(self, motion_mask, bbox):
        if motion_mask is None:
            return 0.0
        h, w = motion_mask.shape[:2]
        x1, y1, x2, y2 = np.asarray(bbox, dtype=int)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        roi = motion_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.count_nonzero(roi)) / max(float(roi.size), 1.0)

    def _evaluate(self, obj, state, now, global_motion_ratio):
        model = obj.get('model', '')
        rules = self._rules_for(model)
        duration = max(0.0, now - state['first_seen'])
        avg_conf = float(np.mean(state['conf_history'])) if state['conf_history'] else float(obj.get('confidence', 0.0))
        avg_motion = float(np.mean(state['motion_history'])) if state['motion_history'] else 0.0
        motion_frames = sum(1 for v in state['motion_history'] if v >= rules.get('min_motion_ratio', 0.0))
        avg_area_ratio = float(np.mean(state['area_history'])) if state['area_history'] else 0.0
        avg_width = float(np.mean(state['width_history'])) if state['width_history'] else 0.0
        avg_height = float(np.mean(state['height_history'])) if state['height_history'] else 0.0

        base_ok = (
            obj.get('missed', 0) <= self.max_missed_frames and
            state['seen_frames'] >= int(rules.get('min_seen_frames', 1)) and
            duration >= float(rules.get('min_duration', 0.0)) and
            avg_conf >= float(rules.get('min_avg_confidence', 0.0)) and
            avg_area_ratio >= float(rules.get('min_area_ratio', 0.0)) and
            avg_width >= float(rules.get('min_width', 1)) and
            avg_height >= float(rules.get('min_height', 1))
        )

        if model == 'fire':
            local_motion_gate = max(
                float(rules.get('min_motion_ratio', 0.0)),
                global_motion_ratio * float(rules.get('local_motion_vs_global', 0.0))
            )
            motion_frames = sum(1 for v in state['motion_history'] if v >= local_motion_gate)
            fusion_evidence = any(
                obj.get(flag) for flag in (
                    'fusion_boosted',
                    'smoke_boosted',
                    'fusion_synthetic',
                    'synthetic_validated',
                    'persistence_active',
                )
            )
            motion_ok = (
                motion_frames >= int(rules.get('min_motion_frames', 1)) and
                avg_motion >= local_motion_gate
            )
            shake_rejected = (
                global_motion_ratio >= self.global_motion_threshold and
                avg_motion < local_motion_gate
            )
            visual_valid = base_ok and not shake_rejected and (
                motion_ok or (rules.get('allow_fusion_evidence', True) and fusion_evidence)
            )
            alarm_gate = (
                motion_frames >= int(rules.get('alarm_min_motion_frames', rules.get('min_motion_frames', 1))) or
                bool(obj.get('synthetic_validated')) or
                bool(obj.get('smoke_boosted')) or
                bool(obj.get('persistence_active'))
            )
            alarm_valid = bool(visual_valid and alarm_gate)
        else:
            visual_valid = base_ok and obj.get('missed', 0) == 0
            alarm_valid = bool(
                visual_valid and
                state['seen_frames'] >= int(rules.get('alarm_min_seen_frames', rules.get('min_seen_frames', 1)))
            )

        if obj.get('missed', 0) > 0 and state.get('visual_valid', False):
            visual_valid = True
        if obj.get('missed', 0) > 0 and state.get('alarm_valid', False):
            alarm_valid = False

        state['visual_valid'] = visual_valid
        state['alarm_valid'] = alarm_valid
        state['duration'] = duration
        return {
            'stream_visual_valid': visual_valid,
            'stream_alarm_valid': alarm_valid,
            'stream_monitor_valid': visual_valid if model in BAG_MODEL_CLASSES else False,
            'stream_avg_confidence': avg_conf,
            'stream_avg_motion': avg_motion,
            'stream_motion_frames': motion_frames,
            'stream_duration': duration,
            'stream_global_motion': global_motion_ratio,
            'stream_area_ratio': avg_area_ratio,
        }

    def update(self, frame, tracked_objects, now):
        if not self.enabled:
            enriched = []
            for obj in tracked_objects:
                clone = dict(obj)
                clone['stream_visual_valid'] = True
                clone['stream_alarm_valid'] = obj.get('state') == 'confirmed' and obj.get('missed', 0) == 0
                clone['stream_monitor_valid'] = clone['stream_visual_valid'] if clone.get('model') in BAG_MODEL_CLASSES else False
                enriched.append(clone)
            return enriched

        _, motion_mask, global_motion_ratio = self._prepare_motion_mask(frame)
        current_keys = set()
        enriched = []
        frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)

        for obj in tracked_objects:
            clone = dict(obj)
            key = self._track_key(clone)
            if key[1] is None:
                enriched.append(clone)
                continue

            current_keys.add(key)
            state = self.track_states.setdefault(key, {
                'first_seen': now,
                'last_seen': now,
                'seen_frames': 0,
                'conf_history': deque(maxlen=self.history_size),
                'motion_history': deque(maxlen=self.history_size),
                'area_history': deque(maxlen=self.history_size),
                'width_history': deque(maxlen=self.history_size),
                'height_history': deque(maxlen=self.history_size),
                'visual_valid': False,
                'alarm_valid': False,
            })

            bbox = np.asarray(clone['bbox'], dtype=float)
            width = max(1.0, bbox[2] - bbox[0])
            height = max(1.0, bbox[3] - bbox[1])
            area_ratio = (width * height) / frame_area

            if clone.get('missed', 0) == 0:
                state['seen_frames'] += 1
                state['last_seen'] = now
                state['conf_history'].append(float(clone.get('confidence', 0.0)))
                state['motion_history'].append(self._local_motion_ratio(motion_mask, bbox))
                state['area_history'].append(area_ratio)
                state['width_history'].append(width)
                state['height_history'].append(height)

            clone.update(self._evaluate(clone, state, now, global_motion_ratio))
            enriched.append(clone)

        stale_keys = [
            key for key, state in self.track_states.items()
            if key not in current_keys and (now - state.get('last_seen', now)) > 1.5
        ]
        for key in stale_keys:
            del self.track_states[key]

        return enriched


def initialize_detectors(fire_model="best.pt", knife_model="knife_best.pt", bag_model="bestSON.pt",
                         classifier_model="fire_classifier.pt", train_model="intrusion_best.pt"):
    """Initialize all detection models and their trackers."""
    global detectors, trackers, cascade, fusion, bag_cascade, knife_cascade, bag_monitor, bag_abandonment_config
    global bag_detection_config, knife_detection_config, fire_postfilter_config
    global camera_runtime_config, stream_validation_config
    global camera_validator, phone_validator, config
    global BAG_MODEL_CLASSES, MODEL_CLASS_GROUPS

    config = ConfigLoader("config.yaml")
    fire_config = config.get_fire_config()
    bag_abandonment_config = config.get('bag_abandonment', {}) or {}
    bag_detection_config = config.get('bag_detection', {}) or {}
    knife_detection_config = config.get('knife_detection', {}) or {}
    fire_postfilter_config = fire_config.get('postfilter', {}) or {}
    camera_runtime_config = config.get('camera_runtime', {}) or {}
    stream_validation_config = config.get('stream_validation', {}) or {}
    BAG_MODEL_CLASSES = _normalize_class_names(
        (bag_detection_config or {}).get('allowed_classes'),
        DEFAULT_BAG_MODEL_CLASSES,
    )
    MODEL_CLASS_GROUPS['bag'] = BAG_MODEL_CLASSES
    bag_monitor = AbandonedBagMonitor(config=bag_abandonment_config)
    camera_validator = StreamDetectionValidator(config=stream_validation_config)
    phone_validator = StreamDetectionValidator(config=stream_validation_config)

    # Fire detector
    fire_conf = fire_config.get('inference_threshold', fire_config['confidence_threshold'])
    if Path(fire_model).exists():
        detectors['fire'] = FireDetector(
            model_path=fire_model,
            conf_threshold=fire_conf,
            iou_threshold=fire_config['iou_threshold']
        )
        trackers['fire'] = ObjectTracker(max_age=TRACKING_MAX_AGE, min_hits=TRACKING_MIN_HITS)
        print(f"Fire detection model loaded! (conf: {fire_conf})")

        # Cascade: if classifier weights exist, enable two-stage detection for fire
        fire_cascade_cfg = fire_config.get('cascade', {}) or {}
        if Path(classifier_model).exists():
            cascade = CascadeFireDetector(
                yolo_detector=detectors['fire'],
                classifier_path=classifier_model,
                classifier_model=str(fire_cascade_cfg.get('classifier_model', 'efficientnet_b0')),
                alpha=float(fire_cascade_cfg.get('alpha', 0.4)),
                fusion_threshold=float(fire_cascade_cfg.get('fusion_threshold', 0.5)),
                min_classifier_prob=float(fire_cascade_cfg.get('min_classifier_prob', 0.4)),
                crop_padding=float(fire_cascade_cfg.get('crop_padding', 0.15)),
            )
            print(
                "Cascade classifier loaded! "
                f"({classifier_model}, alpha={cascade.alpha}, "
                f"fusion={cascade.fusion_threshold}, cls_min={cascade.min_classifier_prob})"
            )
        else:
            print(f"No classifier at {classifier_model} — running YOLO-only for fire")
    else:
        print(f"WARNING: Fire model not found at {fire_model}")

    # Decision Fusion: brightness + motion yardımcı karar
    fusion_cfg = fire_config.get('decision_fusion', {})
    if fusion_cfg.get('enabled', True):
        fusion = DecisionFusion(config=fusion_cfg)
        print(f"Decision Fusion enabled! (debug={fusion_cfg.get('debug', False)})")
    else:
        print("Decision Fusion disabled in config")

    # Knife detector
    knife_conf = float((knife_detection_config or {}).get('confidence_threshold', 0.30))
    if Path(knife_model).exists():
        detectors['knife'] = FireDetector(
            model_path=knife_model,
            conf_threshold=knife_conf,
            iou_threshold=0.5
        )
        trackers['knife'] = ObjectTracker(max_age=TRACKING_MAX_AGE, min_hits=TRACKING_MIN_HITS)
        print(f"Knife detection model loaded! (conf: {knife_conf})")

        knife_classifier_path = str((knife_detection_config or {}).get('classifier_path', 'knife_classifier.pt'))
        use_knife_classifier = bool((knife_detection_config or {}).get('use_classifier', False))
        if use_knife_classifier and Path(knife_classifier_path).exists():
            knife_cascade = KnifeCascadeDetector(
                yolo_detector=detectors['knife'],
                classifier_path=knife_classifier_path,
                classifier_model=str((knife_detection_config or {}).get('classifier_model', 'efficientnet_b0')),
                alpha=float((knife_detection_config or {}).get('classifier_alpha', 0.35)),
                fusion_threshold=float((knife_detection_config or {}).get('classifier_fusion_threshold', 0.50)),
                min_classifier_prob=float((knife_detection_config or {}).get('classifier_min_probability', 0.65)),
                crop_padding=float((knife_detection_config or {}).get('classifier_crop_padding', 0.10)),
                target_class=str((knife_detection_config or {}).get('classifier_target_class', 'knife')),
            )
            print(f"Knife cascade classifier loaded! ({knife_classifier_path})")
        else:
            knife_cascade = None
            print("Knife classifier disabled - running YOLO-only for knife")
    else:
        print(f"WARNING: Knife model not found at {knife_model}")

    # Bag detector
    bag_model = _resolve_bag_model_path(bag_model)
    bag_conf = float((bag_detection_config or {}).get('confidence_threshold', 0.45))
    bag_target_classes = _ordered_bag_classes(BAG_MODEL_CLASSES)
    if Path(bag_model).exists():
        detectors['bag'] = FireDetector(
            model_path=bag_model,
            conf_threshold=bag_conf,
            iou_threshold=0.5
        )
        trackers['bag'] = ObjectTracker(
            max_age=int((bag_detection_config or {}).get('tracker_max_age', 8)),
            min_hits=int((bag_detection_config or {}).get('tracker_min_hits', 1))
        )
        print(f"Bag detection model loaded from {bag_model}! (conf: {bag_conf})")

        bag_classifier_path = str((bag_detection_config or {}).get('classifier_path', 'bag_classifier.pt'))
        use_bag_classifier = bool((bag_detection_config or {}).get('use_classifier', False))
        if use_bag_classifier and Path(bag_classifier_path).exists():
            bag_cascade = BagCascadeDetector(
                yolo_detector=detectors['bag'],
                classifier_path=bag_classifier_path,
                classifier_model=str((bag_detection_config or {}).get('classifier_model', 'efficientnet_b0')),
                alpha=float((bag_detection_config or {}).get('classifier_alpha', 0.35)),
                fusion_threshold=float((bag_detection_config or {}).get('classifier_fusion_threshold', 0.45)),
                min_classifier_prob=float((bag_detection_config or {}).get('classifier_min_probability', 0.60)),
                crop_padding=float((bag_detection_config or {}).get('classifier_crop_padding', 0.10)),
                target_classes=bag_target_classes,
            )
            print(f"Bag cascade classifier loaded! ({bag_classifier_path})")
        else:
            bag_cascade = None
            allowed = "/".join(bag_target_classes)
            print(f"Bag classifier disabled - running YOLO-only for {allowed}")
    else:
        print(f"WARNING: Bag model not found at {bag_model}")

    # Train detector
    train_conf = 0.50
    if Path(train_model).exists():
        detectors['train'] = FireDetector(
            model_path=train_model,
            conf_threshold=train_conf,
            iou_threshold=0.5
        )
        trackers['train'] = ObjectTracker(max_age=TRACKING_MAX_AGE, min_hits=TRACKING_MIN_HITS)
        print(f"Train detection model loaded! (conf: {train_conf})")
    else:
        print(f"WARNING: Train model not found at {train_model}")

    print(f"Active models: {list(detectors.keys())}")


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


def _fire_detect_with_fusion(image, fusion_engine=None, postfilter_override=None):
    """
    Fire detection pipeline: YOLO/Cascade → Decision Fusion.
    Fusion, YOLO'nun düşük confidence verdiği parlak+hareketli ateşleri boost eder.
    """
    if fusion_engine is None:
        fusion_engine = fusion

    # Stage 1: YOLO (veya cascade) detection
    if cascade is not None:
        dets = cascade.detect(image)
    else:
        dets = detectors['fire'].detect(image)

    # Stage 2: Decision Fusion (brightness + motion)
    if fusion_engine is not None:
        result = fusion_engine.process_frame(image, dets)
        dets = result['all_detections']

    # Synthetic detections are a fallback path for missed bright flames.
    # Re-check them with the classifier so local bright objects do not survive.
    if cascade is not None and getattr(cascade, 'cascade_enabled', False):
        validated = []
        frame_height = image.shape[0]
        top_edge_margin = max(2, int(frame_height * 0.01))
        for det in dets:
            if det.get('class_name') == 'fire' and det.get('fusion_synthetic'):
                crop = extract_crop(image, det['bbox'], getattr(cascade, 'crop_padding', 0.15))
                if crop is None:
                    continue
                cls_prob = cascade.classifier.classify_crop(crop)
                bbox = det['bbox'].astype(float)
                if cls_prob < SYNTHETIC_VALIDATION_THRESHOLD:
                    continue
                if bbox[1] <= top_edge_margin and cls_prob < 0.70:
                    continue
                det = dict(det)
                det['classifier_prob'] = cls_prob
                det['synthetic_validated'] = True
                det['confidence'] = max(
                    det['confidence'],
                    min(0.95, 0.35 * det['confidence'] + 0.65 * cls_prob)
                )
            validated.append(det)
        dets = validated

    dets = _filter_fire_false_positives(image, dets, override_cfg=postfilter_override)

    return dets


def _bag_detect_with_classifier(image):
    """Bag/person/luggage detection pipeline: bestSON -> optional crop classifier."""
    if bag_cascade is not None:
        dets = bag_cascade.detect(image)
    else:
        dets = detectors['bag'].detect(image)
    dets = _filter_model_detections('bag', dets, image.shape)
    return dets


def _expand_bbox(box, expand_ratio, image_shape=None):
    """Expand a bbox by a relative ratio."""
    bbox = np.asarray(box, dtype=float)
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * float(expand_ratio)
    pad_y = height * float(expand_ratio)
    expanded = np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=float)
    if image_shape is not None:
        image_h, image_w = image_shape[:2]
        expanded[0] = max(0.0, min(expanded[0], image_w - 1))
        expanded[1] = max(0.0, min(expanded[1], image_h - 1))
        expanded[2] = max(expanded[0] + 1.0, min(expanded[2], image_w))
        expanded[3] = max(expanded[1] + 1.0, min(expanded[3], image_h))
    return expanded


def _point_in_bbox(point, bbox):
    """Return True if point lies inside bbox."""
    x, y = point
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
    return x1 <= x <= x2 and y1 <= y <= y2


def _hand_zones_for_person(person_bbox, image_shape, cfg):
    """Approximate left/right hand regions from a person bbox."""
    px1, py1, px2, py2 = np.asarray(person_bbox, dtype=float)
    person_w = max(1.0, px2 - px1)
    person_h = max(1.0, py2 - py1)

    arm_band_top = py1 + person_h * float(cfg.get('hand_band_top_ratio', 0.42))
    arm_band_bottom = py1 + person_h * float(cfg.get('hand_band_bottom_ratio', 0.92))
    inner_width = person_w * float(cfg.get('hand_inner_width_ratio', 0.20))
    outer_width = person_w * float(cfg.get('hand_outer_width_ratio', 0.20))

    left_zone = np.array([
        px1 - outer_width,
        arm_band_top,
        px1 + inner_width,
        arm_band_bottom,
    ], dtype=float)
    right_zone = np.array([
        px2 - inner_width,
        arm_band_top,
        px2 + outer_width,
        arm_band_bottom,
    ], dtype=float)

    return [
        _expand_bbox(left_zone, 0.0, image_shape=image_shape),
        _expand_bbox(right_zone, 0.0, image_shape=image_shape),
    ]


def _skin_ratio_near_bbox(image, bbox, expand_ratio):
    """Estimate whether a bbox is touching a hand/skin-colored region."""
    expanded = _expand_bbox(bbox, expand_ratio, image_shape=image.shape).astype(int)
    x1, y1, x2, y2 = expanded.tolist()
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(np.count_nonzero(mask)) / max(float(mask.size), 1.0)


def _detect_person_boxes(image):
    """Use the shared bag/person detector to get person boxes for contextual filtering."""
    if 'bag' not in detectors:
        return []
    try:
        detections = _bag_detect_with_classifier(image)
    except Exception:
        return []
    return [
        np.asarray(det['bbox'], dtype=float)
        for det in detections
        if str(det.get('class_name', '')).lower() == 'person'
    ]


def _filter_knife_detections_for_person_context(image, detections):
    """Keep knife detections only when they are plausibly in a person's hand."""
    cfg = {
        'require_person_context': True,
        'person_expand_ratio': 0.08,
        'person_min_overlap_ratio': 0.08,
        'person_max_hand_distance_ratio': 0.22,
        'hand_band_top_ratio': 0.42,
        'hand_band_bottom_ratio': 0.92,
        'hand_inner_width_ratio': 0.20,
        'hand_outer_width_ratio': 0.20,
        'hand_zone_min_cover': 0.18,
        'allow_skin_fallback': True,
        'skin_expand_ratio': 0.35,
        'skin_min_ratio': 0.10,
    }
    cfg.update(knife_detection_config or {})
    if not bool(cfg.get('require_person_context', False)):
        return detections
    if not detections:
        return detections

    person_boxes = _detect_person_boxes(image)

    kept = []
    for det in detections:
        knife_bbox = np.asarray(det['bbox'], dtype=float)
        knife_center = (
            float((knife_bbox[0] + knife_bbox[2]) / 2.0),
            float((knife_bbox[1] + knife_bbox[3]) / 2.0),
        )
        accepted = False

        for person_bbox in person_boxes:
            expanded_person = _expand_bbox(
                person_bbox,
                cfg.get('person_expand_ratio', 0.12),
                image_shape=image.shape,
            )
            overlap_ratio = _compute_bbox_cover(knife_bbox, expanded_person)

            px1, py1, px2, py2 = person_bbox
            person_w = max(1.0, px2 - px1)
            person_h = max(1.0, py2 - py1)
            person_diag = max(np.sqrt(person_w ** 2 + person_h ** 2), 1.0)
            hand_y = py1 + person_h * 0.58
            left_hand = (px1, hand_y)
            right_hand = (px2, hand_y)
            hand_dist = min(
                np.hypot(knife_center[0] - left_hand[0], knife_center[1] - left_hand[1]),
                np.hypot(knife_center[0] - right_hand[0], knife_center[1] - right_hand[1]),
            ) / person_diag

            hand_zones = _hand_zones_for_person(person_bbox, image.shape, cfg)
            hand_zone_cover = max(_compute_bbox_cover(knife_bbox, hand_zone) for hand_zone in hand_zones)
            center_in_hand_zone = any(_point_in_bbox(knife_center, hand_zone) for hand_zone in hand_zones)

            if (
                _point_in_bbox(knife_center, expanded_person) and
                overlap_ratio >= float(cfg.get('person_min_overlap_ratio', 0.08)) and
                (
                    center_in_hand_zone or
                    hand_zone_cover >= float(cfg.get('hand_zone_min_cover', 0.18)) or
                    hand_dist <= float(cfg.get('person_max_hand_distance_ratio', 0.22))
                )
            ):
                accepted = True
                break

        if accepted:
            kept.append(det)
            continue

        if (
            bool(cfg.get('allow_skin_fallback', True)) and
            _skin_ratio_near_bbox(
                image,
                knife_bbox,
                float(cfg.get('skin_expand_ratio', 0.35)),
            ) >= float(cfg.get('skin_min_ratio', 0.10))
        ):
            kept.append(det)

    return kept


def _knife_detect_with_classifier(image):
    """Knife detection pipeline: YOLO -> optional crop classifier."""
    if knife_cascade is not None:
        dets = knife_cascade.detect(image)
    else:
        dets = detectors['knife'].detect(image)
    return _filter_knife_detections_for_person_context(image, dets)


def run_multi_detect(image, active_models):
    """Run detection with multiple models and merge results (no tracking)."""
    all_detections = []
    for model_name in active_models:
        if model_name in detectors:
            if model_name == 'fire':
                dets = _fire_detect_with_fusion(image)
            elif model_name == 'knife':
                dets = _knife_detect_with_classifier(image)
            elif model_name == 'bag':
                dets = _bag_detect_with_classifier(image)
            else:
                dets = _filter_model_detections(model_name, detectors[model_name].detect(image), image.shape)
            for d in dets:
                d['model'] = _resolve_model_key(model_name, d.get('class_name', ''))
            all_detections.extend(dets)
    return all_detections


def run_multi_track(
    image,
    active_models,
    tracker_dict,
    fire_fusion_engine=None,
    fire_postfilter_override=None,
    inference_max_width=None,
):
    """
    Run detect() + ObjectTracker.update() for each model.
    Uses cascade + decision fusion for fire.
    tracker_dict: {model_name: ObjectTracker}
    """
    all_tracked = []
    infer_image, infer_scale = _resize_for_inference(image, inference_max_width)
    for model_name in active_models:
        if model_name not in detectors or model_name not in tracker_dict:
            continue
        if model_name == 'fire':
            dets = _fire_detect_with_fusion(
                infer_image,
                fusion_engine=fire_fusion_engine,
                postfilter_override=fire_postfilter_override,
            )
        elif model_name == 'knife':
            dets = _knife_detect_with_classifier(infer_image)
        elif model_name == 'bag':
            dets = _bag_detect_with_classifier(infer_image)
        else:
            dets = _filter_model_detections(model_name, detectors[model_name].detect(infer_image), infer_image.shape)
        dets = _scale_detection_bboxes(dets, infer_scale)
        tracked = tracker_dict[model_name].update(dets)

        for t in tracked:
            t['model'] = _resolve_model_key(model_name, t.get('class_name', ''))
        all_tracked.extend(tracked)
    return all_tracked


def run_multi_predict_only(active_models, tracker_dict):
    """
    Advance Kalman filters WITHOUT running detection.
    Use on skipped frames: bbox moves with predicted velocity.
    """
    all_tracked = []
    for model_name in active_models:
        if model_name not in tracker_dict:
            continue
        tracked = tracker_dict[model_name].predict_only()

        for t in tracked:
            t['model'] = _resolve_model_key(model_name, t.get('class_name', ''))
        all_tracked.extend(tracked)
    return all_tracked


def _apply_stream_validation(validator, frame, tracked_objects, now):
    """Filter unstable live-stream tracks before drawing and alarming."""
    if validator is None:
        visual_tracks = list(tracked_objects)
        monitor_tracks = [obj for obj in tracked_objects if obj.get('model') in BAG_MODEL_CLASSES]
        return visual_tracks, monitor_tracks

    validated = validator.update(frame, tracked_objects, now)
    visual_tracks = [obj for obj in validated if obj.get('stream_visual_valid', True)]
    monitor_tracks = [obj for obj in validated if obj.get('stream_monitor_valid', False)]
    return visual_tracks, monitor_tracks


def draw_multi_detections(image, detections):
    """Draw bounding boxes with model-specific colors (for image upload, no tracking)."""
    for det in detections:
        bbox = det['bbox'].astype(int)
        conf = det['confidence']
        model = det.get('model', 'fire')
        color = MODEL_COLORS.get(model, (0, 0, 255))
        label_prefix = MODEL_LABELS.get(model, model.upper())

        # Persistence/smoke bilgisi label'a ekle
        suffix = ''
        if det.get('persistence_active'):
            suffix = ' [HAFIZA]'
            # Persistence: turuncu tonlu kutu
            color = (0, 140, 255)
        elif det.get('smoke_boosted'):
            suffix = ' [DUMAN+]'

        label = f"{label_prefix}: {conf:.2f}{suffix}"

        # Draw bounding box
        thickness = 1 if det.get('persistence_active') else 2
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (bbox[0], bbox[1] - label_height - 10),
                     (bbox[0] + label_width, bbox[1]), color, -1)

        # Draw label text
        cv2.putText(image, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def draw_tracked_detections(image, tracked_objects):
    """
    Draw bounding boxes with confidence (no track ID shown).

    - missed > 2 → don't draw (ghost box prevention)
    - tentative: thin box, dimmed
    - confirmed: solid thick box
    - lost (missed <= 2): thin fading box (Kalman predicted)
    """
    for obj in tracked_objects:
        if not obj.get('stream_visual_valid', True):
            continue

        missed = obj.get('missed', 0)

        # Hayalet kutu engeli: 2 frame'den fazla kayipsa cizme
        if missed > 2:
            continue

        bbox = obj['bbox'].astype(int)
        conf = obj['confidence']
        state = obj.get('state', 'confirmed')
        model = obj.get('model', 'fire')
        color = MODEL_COLORS.get(model, (0, 0, 255))
        label_prefix = MODEL_LABELS.get(model, model.upper())
        if model in ABANDONED_BAG_MODELS and bag_detection_config.get('hide_tentative_tracks', True) and state == 'tentative':
            continue
        if model in ABANDONED_BAG_MODELS:
            if obj.get('bag_alert_abandoned'):
                color = (0, 0, 255)
                label_prefix = 'TERK CANTA'
            elif obj.get('bag_alert_warning'):
                color = (0, 165, 255)
                label_prefix = 'SUPHELI CANTA'
        elif model == 'fire' and obj.get('stream_alarm_valid') and obj.get('track_id') is not None:
            label_prefix = f"{label_prefix} #{obj['track_id']}"

        # Label: sadece isim + confidence (track ID yok)
        label = f"{label_prefix}: {conf:.2f}"
        if model in ABANDONED_BAG_MODELS and obj.get('bag_alert_duration'):
            label += f" | {obj['bag_alert_duration']:.0f}s"
        elif model == 'fire' and obj.get('stream_avg_motion', 0.0) > 0:
            label += f" | M{obj['stream_avg_motion']:.2f}"

        if state == 'tentative':
            box_color = tuple(int(c * 0.5) for c in color)
            thickness = 1
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, thickness)
        elif state == 'lost':
            alpha = max(0.4, 1.0 - missed * 0.2)
            box_color = tuple(int(c * alpha) for c in color)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 1)
        else:
            # Confirmed: solid thick box
            box_color = color
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)

        # Draw label background + text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (bbox[0], bbox[1] - label_height - 10),
                     (bbox[0] + label_width, bbox[1]), box_color, -1)
        cv2.putText(image, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and detection."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get active models from form data
        active_models = request.form.getlist('models')
        if not active_models:
            active_models = ['fire']  # Default

        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Detect with selected models
        detections = run_multi_detect(image, active_models)

        # Draw detections
        output_image = draw_multi_detections(image.copy(), detections)

        # Convert to base64 for web display
        _, buffer = cv2.imencode('.jpg', output_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare results
        threat_detected = len(detections) > 0
        confidence = max([d['confidence'] for d in detections], default=0) if detections else 0

        # Per-model results
        model_results = {}
        result_keys = _display_keys(active_models)
        for key in result_keys:
            model_dets = [d for d in detections if d.get('model') == key]
            model_results[key] = {
                'detected': len(model_dets) > 0,
                'count': len(model_dets),
                'max_confidence': float(max([d['confidence'] for d in model_dets], default=0))
            }

        result = {
            'threat_detected': threat_detected,
            'confidence': float(confidence),
            'num_detections': len(detections),
            'image': f'data:image/jpeg;base64,{img_base64}',
            'active_models': active_models,
            'model_results': model_results,
            'detections': [
                {
                    'confidence': float(d['confidence']),
                    'bbox': [float(x) for x in d['bbox']],
                    'model': d.get('model', 'fire')
                }
                for d in detections
            ]
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and start processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get active models from form data
        active_models = request.form.getlist('models')
        if not active_models:
            active_models = ['fire']

        # Save video to temp file
        suffix = Path(file.filename).suffix or '.mp4'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        file.save(tmp.name)
        tmp.close()

        job_id = str(uuid.uuid4())
        video_jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'processed_frames': 0,
            'total_frames': 0,
            'detection_frames': 0,
            'max_confidence': 0,
            'input_path': tmp.name,
            'active_models': active_models,
        }

        # Process in background thread
        thread = threading.Thread(target=process_video, args=(job_id,))
        thread.daemon = True
        thread.start()

        return jsonify({'job_id': job_id})

    except Exception as e:
        print(f"Error uploading video: {e}")
        return jsonify({'error': str(e)}), 500


def process_video(job_id):
    """Process video frames with multi-model tracking in background."""
    job = video_jobs[job_id]
    input_path = job['input_path']
    active_models = job.get('active_models', ['fire'])

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            job['status'] = 'error'
            job['error'] = 'Video dosyasi acilamadi'
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        job['total_frames'] = total_frames
        job['fps'] = fps
        job['width'] = width
        job['height'] = height

        # Output processed video
        out_path = input_path.rsplit('.', 1)[0] + '_detected.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        job['output_path'] = out_path

        # Create dedicated trackers for this video (separate from camera)
        video_trackers = {}
        for model_name in active_models:
            if model_name in detectors:
                video_trackers[model_name] = ObjectTracker(max_age=TRACKING_MAX_AGE, min_hits=TRACKING_MIN_HITS)
        video_bag_monitor = AbandonedBagMonitor(config=bag_abandonment_config)
        video_validator = StreamDetectionValidator(config=stream_validation_config)

        # Reset fusion state for fresh video
        if fusion is not None:
            fusion.reset()

        frame_idx = 0
        detection_frames = 0
        max_conf = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Her frame detect + track (skip yok, tam kalite)
            tracked = run_multi_track(frame, active_models, video_trackers)
            stream_time = frame_idx / max(float(fps), 1.0)
            visual_tracks, monitor_tracks = _apply_stream_validation(video_validator, frame, tracked, stream_time)
            _apply_bag_monitor(video_bag_monitor, monitor_tracks, stream_time)
            output_frame = draw_tracked_detections(frame.copy(), visual_tracks)

            # Only count alarm-worthy confirmed tracks
            active_dets = [
                t for t in visual_tracks
                if t.get('stream_alarm_valid', False)
            ]
            alarm_dets = [t for t in active_dets if _is_alarm_object(t)]
            if len(alarm_dets) > 0:
                detection_frames += 1
                conf = max(d['confidence'] for d in alarm_dets)
                if conf > max_conf:
                    max_conf = conf

                # Show status per detected model
                y_offset = 30
                for key in _display_keys(active_models):
                    model_dets = [d for d in alarm_dets if d.get('model') == key]
                    if model_dets:
                        color = MODEL_COLORS.get(key, (0, 0, 255))
                        if key in ABANDONED_BAG_MODELS:
                            if any(d.get('bag_alert_abandoned') for d in model_dets):
                                color = (0, 0, 255)
                            elif any(d.get('bag_alert_warning') for d in model_dets):
                                color = (0, 165, 255)
                        status_text = _alarm_text_for_objects(key, model_dets)
                        if not status_text:
                            continue
                        cv2.putText(output_frame, status_text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        y_offset += 30

            out.write(output_frame)

            job['processed_frames'] = frame_idx
            job['progress'] = int((frame_idx / max(total_frames, 1)) * 100)
            job['detection_frames'] = detection_frames
            job['max_confidence'] = max_conf

        cap.release()
        out.release()

        # Clean up input temp file
        try:
            os.unlink(input_path)
        except OSError:
            pass

        job['status'] = 'done'
        job['progress'] = 100
        print(f"Video processing done: {frame_idx} frames, {detection_frames} detections")

    except Exception as e:
        import traceback
        traceback.print_exc()
        job['status'] = 'error'
        job['error'] = str(e)
        print(f"Video processing error: {e}")


@app.route('/video_status/<job_id>')
def video_status(job_id):
    """Return video processing progress."""
    job = video_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'processed_frames': job.get('processed_frames', 0),
        'total_frames': job.get('total_frames', 0),
        'detection_frames': job.get('detection_frames', 0),
        'max_confidence': job.get('max_confidence', 0),
        'active_models': job.get('active_models', ['fire']),
        'error': job.get('error'),
    })


def generate_video_frames(job_id):
    """Stream processed video as MJPEG."""
    job = video_jobs.get(job_id)
    if not job or 'output_path' not in job:
        return

    # Wait until processing is done
    while job['status'] == 'processing':
        import time
        time.sleep(0.5)

    output_path = job['output_path']
    cap = cv2.VideoCapture(output_path)
    fps = job.get('fps', 25)
    delay = 1.0 / fps

    import time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(delay)

    cap.release()

    # Clean up output file
    try:
        os.unlink(output_path)
    except OSError:
        pass


@app.route('/video_feed/<job_id>')
def video_feed(job_id):
    """Stream processed video."""
    return Response(generate_video_frames(job_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_camera_frames(active_models):
    """Generate camera frames with persistent object tracking."""
    global camera, bag_monitor, camera_validator

    runtime_cfg = {
        'camera_index': 0,
        'backend': 'dshow',
        'detect_every_n_frames': 2,
        'inference_max_width': 960,
        'jpeg_quality': 72,
        'fire_postfilter': {
            'reject_synthetic': True,
            'minimum_fire_quality_score': 0.70,
            'auxiliary_object_veto': {
                'enabled': False,
            },
        },
    }
    runtime_cfg = _merge_nested_dicts(runtime_cfg, camera_runtime_config or {})
    detect_every_n_frames = max(1, int(runtime_cfg.get('detect_every_n_frames', 2)))
    inference_max_width = int(runtime_cfg.get('inference_max_width', 960))
    jpeg_quality = int(runtime_cfg.get('jpeg_quality', 72))
    fire_postfilter_override = runtime_cfg.get('fire_postfilter', {}) or {}
    frame_idx = 0

    camera = _open_live_camera(
        index=int(runtime_cfg.get('camera_index', 0)),
        backend_name=runtime_cfg.get('backend', 'dshow'),
    )
    if camera is None:
        print("Error: Could not open camera")
        return

    # Reset trackers and fusion for fresh camera session
    for model_name in active_models:
        if model_name in trackers:
            trackers[model_name].reset()
    if fusion is not None:
        fusion.reset()
    if bag_monitor is not None:
        bag_monitor.reset()
    if camera_validator is not None:
        camera_validator.reset()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame_idx += 1

            should_detect = (frame_idx == 1) or ((frame_idx - 1) % detect_every_n_frames == 0)
            if should_detect:
                tracked_objects = run_multi_track(
                    frame,
                    active_models,
                    trackers,
                    fire_postfilter_override=fire_postfilter_override,
                    inference_max_width=inference_max_width,
                )
            else:
                tracked_objects = run_multi_predict_only(active_models, trackers)
            now = time.monotonic()
            visual_tracks, monitor_tracks = _apply_stream_validation(camera_validator, frame, tracked_objects, now)
            bag_info = _apply_bag_monitor(bag_monitor, monitor_tracks, now)

            # Draw tracked detections with IDs and state
            output_frame = draw_tracked_detections(frame.copy(), visual_tracks)

            # ALARM: Only for CONFIRMED tracks (3+ consecutive frames)
            confirmed = [t for t in visual_tracks if t.get('stream_alarm_valid', False)]
            alarm_objects = [t for t in confirmed if _is_alarm_object(t)]
            threat_detected = len(alarm_objects) > 0

            y_offset = 30
            if threat_detected:
                for key in _display_keys(active_models):
                    model_dets = [d for d in alarm_objects if d.get('model') == key]
                    if model_dets:
                        color = MODEL_COLORS.get(key, (0, 0, 255))
                        if key in ABANDONED_BAG_MODELS:
                            if any(d.get('bag_alert_abandoned') for d in model_dets):
                                color = (0, 0, 255)
                            elif any(d.get('bag_alert_warning') for d in model_dets):
                                color = (0, 165, 255)
                        mc = max(d['confidence'] for d in model_dets)
                        status_text = _alarm_text_for_objects(key, model_dets)
                        if not status_text:
                            continue
                        cv2.putText(output_frame, f"{status_text} ({mc:.2f})", (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        y_offset += 30
            else:
                cv2.putText(output_frame, "Tehdit Yok", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show track counts by state
            n_confirmed = len([t for t in visual_tracks if t.get('state') == 'confirmed'])
            n_tentative = len([t for t in tracked_objects if t.get('state') == 'tentative'])
            n_lost = len([t for t in tracked_objects if t.get('state') == 'lost'])
            if n_confirmed + n_tentative + n_lost > 0:
                status = f"Takip: {n_confirmed} onaylandi"
                if n_tentative > 0:
                    status += f", {n_tentative} beklemede"
                if n_lost > 0:
                    status += f", {n_lost} kayip"
                cv2.putText(output_frame, status, (10, y_offset + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 20

            if 'bag' in active_models and (bag_info['bag_count'] > 0 or bag_info['person_count'] > 0):
                bag_status = f"Kisi: {bag_info['person_count']} | Canta: {bag_info['bag_count']}"
                if bag_info['abandoned_count'] > 0:
                    bag_status += f" | Terk: {bag_info['abandoned_count']}"
                elif bag_info['warning_count'] > 0:
                    bag_status += f" | Supheli: {bag_info['warning_count']}"
                cv2.putText(output_frame, bag_status, (10, y_offset + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20

            # Fusion durumu: persistence / smoke bilgisi
            if fusion is not None and 'fire' in active_models:
                fusion_info = []
                if fusion.fire_history:
                    fusion_info.append(f"Yangin Hafiza: {len(fusion.fire_history)} bolge")
                if fusion.smoke_enabled and hasattr(fusion, '_last_smoke_ratio') and fusion._last_smoke_ratio > fusion.smoke_ratio_threshold:
                    fusion_info.append("DUMAN ALGILANDI")
                if fusion_info:
                    cv2.putText(output_frame, " | ".join(fusion_info), (10, y_offset + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        if camera:
            camera.release()


@app.route('/camera_feed')
def camera_feed():
    """Video streaming route."""
    models = request.args.get('models', 'fire')
    active_models = [m.strip() for m in models.split(',') if m.strip()]
    if not active_models:
        active_models = ['fire']
    return Response(generate_camera_frames(active_models),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera feed."""
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})


# --- Phone Camera: client-side getUserMedia → server-side detection ---
phone_trackers = {}   # Separate trackers for phone camera stream
phone_fusion = None   # Separate fusion state for phone camera
phone_bag_monitor = None


@app.route('/phone_camera_start', methods=['POST'])
def phone_camera_start():
    """Initialize trackers and fusion for phone camera session."""
    global phone_trackers, phone_fusion, phone_bag_monitor, phone_validator

    data = request.get_json() or {}
    active_models = data.get('models', ['fire'])

    # Create fresh trackers for phone stream
    phone_trackers.clear()
    for model_name in active_models:
        if model_name in detectors:
            phone_trackers[model_name] = ObjectTracker(max_age=TRACKING_MAX_AGE, min_hits=TRACKING_MIN_HITS)

    # Create fresh fusion for phone stream
    if config:
        fire_config = config.get_fire_config()
        fusion_cfg = fire_config.get('decision_fusion', {})
        if fusion_cfg.get('enabled', True):
            phone_fusion = DecisionFusion(config=fusion_cfg)
    phone_bag_monitor = AbandonedBagMonitor(config=bag_abandonment_config)
    phone_validator = StreamDetectionValidator(config=stream_validation_config)

    return jsonify({'status': 'ok', 'models': active_models})


@app.route('/phone_camera_stop', methods=['POST'])
def phone_camera_stop():
    """Clean up phone camera session."""
    global phone_trackers, phone_fusion, phone_bag_monitor, phone_validator
    phone_trackers.clear()
    phone_fusion = None
    phone_bag_monitor = None
    phone_validator = None
    return jsonify({'status': 'stopped'})


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """
    Receive a frame from phone camera, run detection+tracking,
    return processed JPEG directly (no JSON/base64 overhead).

    Accepts:
        FormData with 'frame' (JPEG blob) + 'models' (comma-separated string)
    Returns:
        Raw JPEG image with bounding boxes drawn
    """
    global phone_fusion, phone_bag_monitor, phone_validator

    try:
        # Accept FormData (binary blob — fast)
        if 'frame' in request.files:
            file = request.files['frame']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            models_str = request.form.get('models', 'fire')
            active_models = [m.strip() for m in models_str.split(',') if m.strip()]
        else:
            return Response(b'No frame', status=400, mimetype='text/plain')

        if frame is None:
            return Response(b'Invalid image', status=400, mimetype='text/plain')

        # Run detection + tracking with phone-specific trackers
        all_tracked = []
        for model_name in active_models:
            if model_name not in detectors:
                continue

            if model_name not in phone_trackers:
                phone_trackers[model_name] = ObjectTracker(max_age=TRACKING_MAX_AGE, min_hits=TRACKING_MIN_HITS)

            if model_name == 'fire':
                dets = _fire_detect_with_fusion(frame, fusion_engine=phone_fusion)
            elif model_name == 'knife':
                dets = _knife_detect_with_classifier(frame)
            elif model_name == 'bag':
                dets = _bag_detect_with_classifier(frame)
            else:
                dets = _filter_model_detections(model_name, detectors[model_name].detect(frame), frame.shape)

            tracked = phone_trackers[model_name].update(dets)
            for t in tracked:
                t['model'] = _resolve_model_key(model_name, t.get('class_name', ''))
            all_tracked.extend(tracked)

        # Draw detections on frame
        now = time.monotonic()
        visual_tracks, monitor_tracks = _apply_stream_validation(phone_validator, frame, all_tracked, now)
        bag_info = _apply_bag_monitor(phone_bag_monitor, monitor_tracks, now)
        output_frame = draw_tracked_detections(frame.copy(), visual_tracks)

        # Alarm text
        confirmed = [t for t in visual_tracks if t.get('stream_alarm_valid', False)]
        alarm_objects = [t for t in confirmed if _is_alarm_object(t)]
        if alarm_objects:
            y_offset = 30
            for key in _display_keys(active_models):
                model_dets = [d for d in alarm_objects if d.get('model') == key]
                if model_dets:
                    color = MODEL_COLORS.get(key, (0, 0, 255))
                    if key in ABANDONED_BAG_MODELS:
                        if any(d.get('bag_alert_abandoned') for d in model_dets):
                            color = (0, 0, 255)
                        elif any(d.get('bag_alert_warning') for d in model_dets):
                            color = (0, 165, 255)
                    mc = max(d['confidence'] for d in model_dets)
                    status_text = _alarm_text_for_objects(key, model_dets)
                    if not status_text:
                        continue
                    cv2.putText(output_frame, f"{status_text} ({mc:.2f})", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 30
        else:
            cv2.putText(output_frame, "Tehdit Yok", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if 'bag' in active_models and (bag_info['bag_count'] > 0 or bag_info['person_count'] > 0):
            bag_status = f"Kisi: {bag_info['person_count']} | Canta: {bag_info['bag_count']}"
            if bag_info['abandoned_count'] > 0:
                bag_status += f" | Terk: {bag_info['abandoned_count']}"
            elif bag_info['warning_count'] > 0:
                bag_status += f" | Supheli: {bag_info['warning_count']}"
            cv2.putText(output_frame, bag_status, (10, frame.shape[0] - 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Return raw JPEG (no JSON/base64 overhead)
        _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        print(f"Error in detect_frame: {e}")
        return Response(str(e).encode(), status=500, mimetype='text/plain')


@app.route('/get_settings')
def get_settings():
    """Return current confidence/iou settings for each model."""
    settings = {}
    for name, det in detectors.items():
        settings[name] = {
            'confidence': det.conf_threshold,
            'iou': det.iou_threshold,
        }
    return jsonify(settings)


@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update confidence/iou thresholds for models at runtime."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    updated = {}
    for model_name, params in data.items():
        if model_name not in detectors:
            continue
        det = detectors[model_name]
        if 'confidence' in params:
            val = float(params['confidence'])
            val = max(0.01, min(1.0, val))
            det.conf_threshold = val
            det.model.conf = val
        if 'iou' in params:
            val = float(params['iou'])
            val = max(0.01, min(1.0, val))
            det.iou_threshold = val
            det.model.iou = val
        updated[model_name] = {
            'confidence': det.conf_threshold,
            'iou': det.iou_threshold,
        }

    return jsonify({'status': 'ok', 'updated': updated})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Model Detection Web Application')
    parser.add_argument('--fire-model', type=str,
                       default='best.pt',
                       help='Path to fire detection model weights')
    parser.add_argument('--knife-model', type=str,
                       default='knife_best.pt',
                       help='Path to knife detection model weights')
    parser.add_argument('--bag-model', type=str,
                       default='bestSON.pt',
                       help='Path to bag detection model weights')
    parser.add_argument('--train-model', type=str,
                       default='intrusion_best.pt',
                       help='Path to train detection model weights')
    parser.add_argument('--classifier', type=str,
                       default='fire_classifier.pt',
                       help='Path to fire classifier weights (cascade stage 2)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port to run web server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to run web server (0.0.0.0 for network access)')
    parser.add_argument('--ssl', action='store_true',
                       help='Enable HTTPS (required for phone camera over network)')

    args = parser.parse_args()

    # Initialize detectors
    print("Initializing detection models...")
    initialize_detectors(fire_model=args.fire_model, knife_model=args.knife_model,
                         bag_model=args.bag_model, classifier_model=args.classifier,
                         train_model=args.train_model)

    # SSL context for phone camera (getUserMedia requires HTTPS on non-localhost)
    ssl_ctx = None
    protocol = 'http'
    if args.ssl:
        try:
            ssl_ctx = 'adhoc'
            protocol = 'https'
            print("SSL enabled (adhoc certificate)")
        except Exception:
            print("WARNING: SSL failed, running without HTTPS")
            print("Install pyopenssl: pip install pyopenssl")

    # Run app
    print(f"\nStarting web application...")
    print(f"Open your browser and go to: {protocol}://{args.host}:{args.port}")
    if args.host == '0.0.0.0':
        import socket
        try:
            hostname = socket.gethostbyname(socket.gethostname())
            print(f"Network access: {protocol}://{hostname}:{args.port}")
            print(f"Telefondan baglanmak icin yukaridaki IP'yi kullanin")
        except socket.error:
            pass
    if not args.ssl:
        print("\nNOT: Telefon kamerasi icin --ssl parametresi gereklidir")
        print("  python web_app.py --ssl")
    print(f"\nPress Ctrl+C to stop the server\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True, ssl_context=ssl_ctx)
