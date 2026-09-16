"""
Edge Object Detection wrapper using Ultralytics YOLOv8.
Performs real-time person detection, non-maximum suppression (NMS),
bounding box extraction, and confidence scoring.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from ultralytics import YOLO


class EdgeDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.45, target_classes: List[int] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        # Default target class: 0 (person)
        self.target_classes = target_classes if target_classes is not None else [0]

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Executes forward detection on a video frame.
        Returns list of detections with bounding box (x1, y1, x2, y2), centroid, confidence, and class_id.
        """
        results = self.model(frame, conf=self.conf_thresh, verbose=False, device=self.device)
        detections = []

        if len(results) == 0 or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                    "confidence": conf,
                    "class_id": cls_id
                })

        return detections
