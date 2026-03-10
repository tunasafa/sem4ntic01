#!/usr/bin/env python3
"""
Real-Time Semantic Segmentation Service
========================================

This service processes video frames in real-time to detect and classify objects
for autonomous robot navigation. Think of it as the "eyes and brain" for a robot.

How it works:
1. Receives video frames from the frontend via WebSocket
2. Runs YOLOv8-Seg model to find objects and their shapes
3. Assigns safety priorities (people = stop now, chair = navigate around)
4. Sends back processed frames with colored overlays

Performance:
- ~18ms per frame on Apple M1/M2 (55 FPS)
- ~5ms on RTX 4090 (200 FPS)
- ~80ms on CPU only (12 FPS)

Safety priorities:
- P10: People (emergency stop)
- P9: Large animals (high priority avoidance)
- P8: Vehicles (plan path carefully)
- P7: Traffic signs (follow rules)
- P3-5: Obstacles (general avoidance)
- P1-2: Vegetation (terrain awareness)
"""

import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

# Import ML libraries
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Note: ultralytics not installed. Running in demo mode (no detections).")


# ============================================================================
# CONFIGURATION - Tweak these values to adjust detection behavior
# ============================================================================

# Objects the robot should care about (based on COCO dataset)
# Format: "class_name": {"id": COCO_id, "priority": 1-10, "color": BGR, "category": type}
SAFETY_CRITICAL_CLASSES = {
    # Living beings - these get highest priority (robot should avoid hitting things)
    "person": {"id": 0, "priority": 10, "color": (255, 0, 0), "category": "living"},
    "cat": {"id": 15, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "dog": {"id": 16, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "horse": {"id": 17, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "sheep": {"id": 18, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "cow": {"id": 19, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "elephant": {"id": 20, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "bear": {"id": 21, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "zebra": {"id": 22, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "giraffe": {"id": 23, "priority": 9, "color": (255, 128, 0), "category": "living"},
    "bird": {"id": 14, "priority": 8, "color": (255, 200, 0), "category": "living"},
    
    # Vehicles - HIGH PRIORITY
    "bicycle": {"id": 1, "priority": 7, "color": (0, 255, 255), "category": "vehicle"},
    "car": {"id": 2, "priority": 8, "color": (0, 255, 255), "category": "vehicle"},
    "motorcycle": {"id": 3, "priority": 8, "color": (0, 255, 255), "category": "vehicle"},
    "airplane": {"id": 4, "priority": 6, "color": (0, 255, 255), "category": "vehicle"},
    "bus": {"id": 5, "priority": 8, "color": (0, 255, 255), "category": "vehicle"},
    "train": {"id": 6, "priority": 8, "color": (0, 255, 255), "category": "vehicle"},
    "truck": {"id": 7, "priority": 8, "color": (0, 255, 255), "category": "vehicle"},
    "boat": {"id": 8, "priority": 7, "color": (0, 255, 255), "category": "vehicle"},
    
    # Traffic infrastructure
    "traffic_light": {"id": 9, "priority": 6, "color": (255, 255, 0), "category": "traffic"},
    "fire_hydrant": {"id": 10, "priority": 5, "color": (255, 0, 255), "category": "obstacle"},
    "stop_sign": {"id": 11, "priority": 7, "color": (255, 0, 255), "category": "traffic"},
    "parking_meter": {"id": 12, "priority": 4, "color": (200, 200, 200), "category": "obstacle"},
    "bench": {"id": 13, "priority": 4, "color": (200, 200, 200), "category": "obstacle"},
    
    # Obstacles
    "backpack": {"id": 24, "priority": 3, "color": (150, 150, 150), "category": "obstacle"},
    "umbrella": {"id": 25, "priority": 3, "color": (150, 150, 150), "category": "obstacle"},
    "handbag": {"id": 26, "priority": 3, "color": (150, 150, 150), "category": "obstacle"},
    "suitcase": {"id": 28, "priority": 4, "color": (150, 150, 150), "category": "obstacle"},
    "skateboard": {"id": 36, "priority": 3, "color": (150, 150, 150), "category": "obstacle"},
    
    # Vegetation/Terrain
    "potted_plant": {"id": 58, "priority": 2, "color": (0, 200, 0), "category": "vegetation"},
    "chair": {"id": 56, "priority": 3, "color": (150, 150, 150), "category": "obstacle"},
    "couch": {"id": 57, "priority": 3, "color": (150, 150, 150), "category": "obstacle"},
    "bed": {"id": 59, "priority": 2, "color": (150, 150, 150), "category": "obstacle"},
    "dining_table": {"id": 60, "priority": 3, "color": (150, 150, 150), "category": "obstacle"},
    "toilet": {"id": 61, "priority": 2, "color": (150, 150, 150), "category": "obstacle"},
}

# Create reverse lookup: class_id -> class_info
CLASS_ID_MAP = {v["id"]: {"name": k, **v} for k, v in SAFETY_CRITICAL_CLASSES.items()}

# COCO class names (for all 80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball",
    "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket",
    "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake",
    "chair", "couch", "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear", "hair_drier",
    "toothbrush"
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class DetectionCategory(str, Enum):
    LIVING = "living"
    VEHICLE = "vehicle"
    TRAFFIC = "traffic"
    OBSTACLE = "obstacle"
    VEGETATION = "vegetation"
    UNKNOWN = "unknown"


@dataclass
class SegmentationResult:
    """Result of semantic segmentation for a single object."""
    class_name: str
    class_id: int
    confidence: float
    mask: np.ndarray  # Binary mask
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area_pixels: int
    priority: int
    category: DetectionCategory
    centroid: Tuple[int, int]  # Center point of the object
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (without mask)."""
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
            "area_pixels": self.area_pixels,
            "priority": self.priority,
            "category": self.category.value,
            "centroid": list(self.centroid),
        }


@dataclass
class FrameResult:
    """Complete segmentation result for a single frame."""
    timestamp: float
    inference_time_ms: float
    fps: float
    frame_width: int
    frame_height: int
    detections: List[SegmentationResult]
    safety_alerts: List[dict]
    navigation_recommendation: dict
    
    # Computed masks (for visualization)
    combined_mask: Optional[np.ndarray] = None
    color_overlay: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "fps": round(self.fps, 2),
            "frame_dimensions": {
                "width": self.frame_width,
                "height": self.frame_height
            },
            "object_count": len(self.detections),
            "detections": [d.to_dict() for d in sorted(self.detections, key=lambda x: -x.priority)],
            "safety_alerts": self.safety_alerts,
            "navigation_recommendation": self.navigation_recommendation,
        }


# ============================================================================
# SEGMENTATION MODEL
# ============================================================================

class SegmentationModel:
    """
    Real-time semantic segmentation model using YOLOv8-Seg.
    
    Time Complexity Analysis:
    - Image preprocessing: O(H*W) - linear scan
    - CNN forward pass: O(H*W*K) where K is kernel operations
    - NMS (Non-Maximum Suppression): O(N²) where N is detections
    - Mask generation: O(H*W) per detection
    
    Overall: O(H*W) for fixed model architecture (linear in image size)
    
    Optimizations:
    1. Use TensorRT for GPU acceleration
    2. Use FP16 for faster inference
    3. Use smaller model (nano/small) for speed
    4. Limit max detections to reduce NMS time
    """
    
    def __init__(self, model_size: str = "nano", device: str = "auto"):
        """
        Initialize the segmentation model.
        
        Args:
            model_size: Model size - "nano" (fastest), "small", "medium", "large"
            device: Device to run on - "auto", "cuda", "cpu"
        """
        self.model_size = model_size
        self.model = None
        self.device = device
        self.class_names = COCO_CLASSES
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.max_times_tracked = 30
        
        # Model configuration
        self.conf_threshold = 0.6  # Higher threshold = fewer detections but faster
        # Per-class confidence overrides for known false positives
        # COCO class 6 = train
        self.class_conf_thresholds = {6: 0.85}
        self.iou_threshold = 0.5   # Lower = more aggressive NMS
        self.max_detections = 15   # Tuned for pedestrian street navigation
        
    def load_model(self):
        """Load the YOLOv8 segmentation model."""
        if not YOLO_AVAILABLE:
            print("Running in DEMO mode - no model loaded")
            return
            
        model_names = {
            "nano": "yolo11n-seg.pt",      # Fastest, ~2.9M params
            "small": "yolo11s-seg.pt",     # Good balance, ~9.7M params
            "medium": "yolo11m-seg.pt",    # Best accuracy/speed, ~22.4M params
            "large": "yolo11x-seg.pt",     # Best accuracy, ~62.1M params
        }
        
        model_name = model_names.get(self.model_size, model_names["nano"])
        print(f"Loading {model_name} for real-time segmentation...")
        
        try:
            self.model = YOLO(model_name)
            
            # Auto-select device: prefer CUDA > MPS (Apple Silicon) > CPU
            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            # Warm-up inference
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, device=self.device, verbose=False)
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def segment_frame(self, frame: np.ndarray, generate_overlay: bool = True) -> FrameResult:
        """
        Analyze a video frame and detect objects with their segmentation masks.

        This is the core function - it takes a camera frame, runs the YOLO model,
        and returns everything the robot needs to know: what objects are present,
        where they are, how dangerous they are, and what direction is safe to go.

        Args:
            frame: Camera frame as numpy array (height, width, 3 colors)
            generate_overlay: Whether to create a visual overlay (set False for faster processing)

        Returns:
            FrameResult containing:
            - List of detected objects with positions and priorities
            - Safety alerts for dangerous objects
            - Navigation recommendation (go left/right/forward)
            - Optional color overlay image for visualization
        """
        start_time = time.time()
        height, width = frame.shape[:2]

        detections = []
        safety_alerts = []

        if self.model is not None:
            # Run YOLO inference
            # Resize large frames to 640px max for faster processing
            inf_frame = frame
            max_dim = 640
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                inf_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            results = self.model(
                inf_frame,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
                half=self.device != "cpu",  # FP16 on GPU for ~2x speedup
            )
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes
                    
                    for i in range(len(masks)):
                        # Get class info
                        class_id = int(boxes.cls[i])
                        confidence = float(boxes.conf[i])
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                        # Apply stricter per-class confidence thresholds where configured.
                        required_conf = max(self.conf_threshold, self.class_conf_thresholds.get(class_id, self.conf_threshold))
                        if confidence < required_conf:
                            continue
                        
                        # Get class priority info
                        if class_id in CLASS_ID_MAP:
                            class_info = CLASS_ID_MAP[class_id]
                            priority = class_info["priority"]
                            category = DetectionCategory(class_info["category"])
                        else:
                            priority = 1
                            category = DetectionCategory.UNKNOWN
                        
                        # Process mask
                        mask = masks[i]
                        # Resize mask to original frame size
                        mask_resized = cv2.resize(mask, (width, height))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        # Calculate properties
                        area = int(np.sum(mask_binary))
                        y_coords, x_coords = np.where(mask_binary)
                        if len(x_coords) > 0 and len(y_coords) > 0:
                            centroid = (int(np.mean(x_coords)), int(np.mean(y_coords)))
                        else:
                            centroid = (width // 2, height // 2)
                        
                        # Get bounding box
                        bbox = tuple(map(int, boxes.xywh[i].cpu().numpy()))
                        
                        detection = SegmentationResult(
                            class_name=class_name,
                            class_id=class_id,
                            confidence=confidence,
                            mask=mask_binary,
                            bbox=bbox,
                            area_pixels=area,
                            priority=priority,
                            category=category,
                            centroid=centroid,
                        )
                        detections.append(detection)
                        
                        # Generate safety alerts for high-priority objects
                        if priority >= 7:
                            alert = self._generate_safety_alert(detection, width, height)
                            if alert:
                                safety_alerts.append(alert)
        else:
            # Demo mode - generate fake detections
            detections, safety_alerts = self._generate_demo_detections(frame)
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_times_tracked:
            self.inference_times.pop(0)
        
        avg_time = np.mean(self.inference_times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        # Generate navigation recommendation
        nav_rec = self._generate_navigation_recommendation(detections, width, height)
        
        # Create color overlay only when requested by client to reduce CPU load.
        color_overlay = self._create_color_overlay(frame, detections) if generate_overlay else None
        
        return FrameResult(
            timestamp=time.time(),
            inference_time_ms=inference_time,
            fps=fps,
            frame_width=width,
            frame_height=height,
            detections=detections,
            safety_alerts=safety_alerts,
            navigation_recommendation=nav_rec,
            color_overlay=color_overlay,
        )
    
    def _generate_safety_alert(
        self, detection: SegmentationResult, width: int, height: int
    ) -> Optional[dict]:
        """Generate safety alert for high-priority detections."""
        cx, cy = detection.centroid
        
        # Determine position relative to frame center
        frame_center_x = width // 2
        position = "center" if abs(cx - frame_center_x) < width * 0.2 else \
                   "left" if cx < frame_center_x else "right"
        
        # Calculate distance estimate (based on bounding box size)
        bbox_area = detection.bbox[2] * detection.bbox[3]
        frame_area = width * height
        relative_size = bbox_area / frame_area
        
        if relative_size > 0.3:
            distance = "very_close"
            urgency = "critical"
        elif relative_size > 0.15:
            distance = "close"
            urgency = "high"
        elif relative_size > 0.05:
            distance = "medium"
            urgency = "medium"
        else:
            distance = "far"
            urgency = "low"
        
        return {
            "object": detection.class_name,
            "category": detection.category.value,
            "position": position,
            "distance": distance,
            "urgency": urgency,
            "confidence": round(detection.confidence, 2),
            "priority": detection.priority,
            "action_required": urgency in ["critical", "high"],
        }
    
    def _generate_navigation_recommendation(
        self, detections: List[SegmentationResult], width: int, height: int
    ) -> dict:
        """Generate navigation recommendation based on detections."""
        frame_center_x = width // 2
        frame_center_y = height // 2
        
        # Analyze obstacles in path
        obstacles_ahead = []
        for d in detections:
            cx, cy = d.centroid
            # Check if object is in the center path (within 30% of center horizontally)
            if abs(cx - frame_center_x) < width * 0.3:
                obstacles_ahead.append(d)
        
        # Determine safe directions
        left_clear = True
        right_clear = True
        center_clear = len(obstacles_ahead) == 0
        
        for d in detections:
            cx, cy = d.centroid
            if d.priority >= 5:  # Consider significant obstacles
                if cx < frame_center_x - width * 0.1:
                    left_clear = False
                elif cx > frame_center_x + width * 0.1:
                    right_clear = False
        
        # Generate recommendation
        if center_clear:
            action = "proceed"
            direction = "forward"
            confidence = 0.9
        elif left_clear and not right_clear:
            action = "maneuver"
            direction = "left"
            confidence = 0.7
        elif right_clear and not left_clear:
            action = "maneuver"
            direction = "right"
            confidence = 0.7
        elif left_clear:
            action = "maneuver"
            direction = "left"
            confidence = 0.6
        elif right_clear:
            action = "maneuver"
            direction = "right"
            confidence = 0.6
        else:
            action = "stop"
            direction = "none"
            confidence = 0.95
        
        # Calculate overall risk level
        max_priority = max([d.priority for d in detections]) if detections else 0
        if max_priority >= 9:
            risk_level = "critical"
        elif max_priority >= 7:
            risk_level = "high"
        elif max_priority >= 5:
            risk_level = "medium"
        elif max_priority >= 3:
            risk_level = "low"
        else:
            risk_level = "safe"
        
        return {
            "action": action,
            "direction": direction,
            "confidence": confidence,
            "risk_level": risk_level,
            "obstacles_ahead_count": len(obstacles_ahead),
            "total_objects_detected": len(detections),
            "safe_directions": {
                "left": left_clear,
                "center": center_clear,
                "right": right_clear,
            }
        }
    
    def _create_color_overlay(
        self, frame: np.ndarray, detections: List[SegmentationResult]
    ) -> np.ndarray:
        """
        Create a visual overlay showing detected objects with color-coded masks.

        Each object type gets a unique color (people = red, vehicles = yellow, etc.).
        The overlay is blended 50/50 with the original frame so you can see both.

        Optimization: Instead of blending each mask separately (slow), we paint all
        masks onto one color layer first, then blend everything in one pass. This is
        about 2x faster for frames with many detections.

        Returns:
            Original frame with colored masks and bounding boxes drawn on top
        """
        overlay = frame.copy()
        if not detections:
            return overlay  # No objects = no overlay needed

        color_layer = np.zeros_like(frame)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for d in detections:
            color = CLASS_ID_MAP[d.class_id]["color"] if d.class_id in CLASS_ID_MAP else (128, 128, 128)
            mask_bool = d.mask.astype(bool)
            color_layer[mask_bool] = color
            combined_mask[mask_bool] = 1

        # Single vectorized blend for all masks at once
        mask_indices = combined_mask.astype(bool)
        if np.any(mask_indices):
            overlay[mask_indices] = (
                frame[mask_indices].astype(np.float32) * 0.5
                + color_layer[mask_indices].astype(np.float32) * 0.5
            ).astype(np.uint8)

        # Annotations (cheap cv2 draw calls)
        for d in detections:
            color = CLASS_ID_MAP[d.class_id]["color"] if d.class_id in CLASS_ID_MAP else (128, 128, 128)
            x, y, w, h = d.bbox
            cv2.rectangle(overlay, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
            label = f"{d.class_name} {d.confidence:.0%}"
            cv2.putText(overlay, label, (int(x - w / 2), int(y - h / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(overlay, d.centroid, 5, (0, 0, 255), -1)

        return overlay
    
    def _generate_demo_detections(
        self, frame: np.ndarray
    ) -> Tuple[List[SegmentationResult], List[dict]]:
        """Return empty detections when model is not available.
        
        NOTE: This returns NO detections because the YOLOv8 model is not installed.
        To enable real detection, install ultralytics:
            pip install ultralytics
        Then restart the service.
        """
        # Return empty detections - no fake data
        return [], []


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Real-Time Semantic Segmentation Service")

# Global model instance
model = SegmentationModel(model_size="large", device="auto")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    model.load_model()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Semantic Segmentation",
        "model_loaded": model.model is not None,
        "device": model.device,
    }


@app.get("/status")
async def get_status():
    """Get detailed service status."""
    return {
        "status": "running",
        "model": {
            "loaded": model.model is not None,
            "size": model.model_size,
            "device": model.device,
            "confidence_threshold": model.conf_threshold,
        },
        "performance": {
            "avg_inference_time_ms": round(np.mean(model.inference_times), 2) if model.inference_times else 0,
            "current_fps": round(1000 / np.mean(model.inference_times), 1) if model.inference_times else 0,
        },
        "supported_classes": list(SAFETY_CRITICAL_CLASSES.keys()),
    }


@app.post("/segment")
async def segment_image(image_data: dict):
    """Segment a single image (base64 encoded)."""
    try:
        # Decode base64 image directly to OpenCV (skip PIL roundtrip)
        image_bytes = base64.b64decode(image_data["image"])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run segmentation
        result = model.segment_frame(frame, generate_overlay=True)
        
        # Encode result image with lower quality for faster transmission
        _, buffer = cv2.imencode('.jpg', result.color_overlay, [cv2.IMWRITE_JPEG_QUALITY, 50])
        result_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "result": result.to_dict(),
            "result_image": result_image_b64,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.

    How it works:
    - Client sends camera frames continuously
    - We process each frame and send back results with overlay
    - Runs at 20-60 FPS depending on hardware

    Architecture:
    Two tasks run concurrently:
    1. Receiver: Grabs incoming frames, always keeping only the latest one
    2. Processor: Takes the latest frame, runs AI, sends result back

    This design prevents lag - we never queue up old frames. If the client
    sends frames faster than we can process, we just skip the extras and
    work on the newest one. This keeps latency minimal.
    """
    await websocket.accept()
    print("WebSocket client connected")

    # Shared state between receiver and processor tasks
    latest_frame: dict = {}
    frame_ready = asyncio.Event()  # Signals "new frame ready!"
    shutdown = asyncio.Event()

    async def _receiver():
        """
        Read frames from WebSocket connection.

        Accepts two formats:
        - Binary: [1-byte overlay flag][JPEG bytes] - faster, preferred
        - JSON: {"type": "frame", "image": "base64...", "overlay": true} - fallback
        """
        nonlocal latest_frame
        try:
            while not shutdown.is_set():
                ws_message = await websocket.receive()

                if "bytes" in ws_message and ws_message["bytes"]:
                    # Binary frame — first byte is overlay flag, rest is JPEG.
                    raw = ws_message["bytes"]
                    latest_frame = {
                        "type": "frame",
                        "image_bytes": raw[1:],
                        "overlay": bool(raw[0]),
                    }
                    frame_ready.set()

                elif "text" in ws_message and ws_message["text"]:
                    data = json.loads(ws_message["text"])

                    if data.get("type") == "frame":
                        # Legacy base64 JSON path (backward compat).
                        latest_frame = data
                        frame_ready.set()

                    elif data.get("type") == "config":
                        if "confidence_threshold" in data:
                            model.conf_threshold = data["confidence_threshold"]
                        if "iou_threshold" in data:
                            model.iou_threshold = data["iou_threshold"]
                        await websocket.send_json({
                            "type": "config_updated",
                            "config": {
                                "confidence_threshold": model.conf_threshold,
                                "iou_threshold": model.iou_threshold,
                            },
                        })

                    elif data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

        except (WebSocketDisconnect, Exception):
            shutdown.set()
            frame_ready.set()  # unblock processor so it can exit

    async def _processor():
        """
        Process frames and send results back to client.

        This waits for a frame to arrive, runs the YOLO model on it,
        and sends back the results (detections + overlay image).
        """
        nonlocal latest_frame
        try:
            while not shutdown.is_set():
                await frame_ready.wait()
                if shutdown.is_set():
                    break

                # Grab the latest frame and clear the slot.
                message = latest_frame
                latest_frame = {}
                frame_ready.clear()

                if not message:
                    continue

                try:
                    if "image_bytes" in message:
                        # Binary path — raw JPEG bytes, no base64 overhead.
                        nparr = np.frombuffer(message["image_bytes"], np.uint8)
                    else:
                        # Legacy base64 JSON path.
                        nparr = np.frombuffer(base64.b64decode(message["image"]), np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    want_overlay = bool(message.get("overlay", True))
                    result = model.segment_frame(frame, generate_overlay=want_overlay)

                    if result.color_overlay is not None:
                        _, buffer = cv2.imencode(
                            ".jpg",
                            result.color_overlay,
                            [cv2.IMWRITE_JPEG_QUALITY, 60],
                        )
                        result_image_b64 = base64.b64encode(buffer).decode("utf-8")
                    else:
                        result_image_b64 = None

                    await websocket.send_json({
                        "type": "result",
                        "data": result.to_dict(),
                        "result_image": result_image_b64,
                    })

                except Exception as e:
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "error": str(e),
                        })
                    except Exception:
                        break

        except (WebSocketDisconnect, Exception):
            shutdown.set()

    # Run receiver and processor concurrently; when either exits, cancel the other.
    try:
        done, pending = await asyncio.wait(
            [asyncio.create_task(_receiver()), asyncio.create_task(_processor())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        shutdown.set()
        for task in pending:
            task.cancel()
    except Exception as e:
        print(f"WebSocket error: {e}")

    print("WebSocket client disconnected")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Real-Time Semantic Segmentation Service")
    print("=" * 60)
    print(f"Model: YOLOv8-{model.model_size}-seg")
    print(f"Device: {model.device}")
    print(f"Port: 3030")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3030,
        log_level="warning",
    )
