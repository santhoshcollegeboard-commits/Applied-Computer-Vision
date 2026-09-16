"""
Structured Alert Dispatcher and Snapshot Archiver.
Regulates event frequency using cooldown timers, logs standardized JSON events,
and stores annotated visual evidence snapshots on disk.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import cv2
import numpy as np


class AlertManager:
    def __init__(self, log_path: str = "alerts.json", snapshot_dir: str = "alert_snapshots", cooldown_sec: float = 5.0):
        self.log_path = log_path
        self.snapshot_dir = snapshot_dir
        self.cooldown_sec = cooldown_sec
        self.last_alerts: Dict[str, float] = {}

        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)

    def dispatch_alert(
        self,
        event_type: str,
        frame: np.ndarray,
        confidence: float,
        target_count: int,
        priority: str = "HIGH",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Dispatches and logs an alert if the cooldown window has elapsed.
        """
        now = time.time()
        if event_type in self.last_alerts:
            if now - self.last_alerts[event_type] < self.cooldown_sec:
                return None  # Suppressed by cooldown

        self.last_alerts[event_type] = now
        timestamp_dt = datetime.now()
        timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
        file_stamp = timestamp_dt.strftime("%Y%m%d_%H%M%S")

        # Save visual snapshot
        snapshot_filename = f"{event_type}_{file_stamp}.jpg"
        snapshot_full_path = os.path.join(self.snapshot_dir, snapshot_filename)
        cv2.imwrite(snapshot_full_path, frame)

        alert_record = {
            "event_type": event_type,
            "timestamp": timestamp_str,
            "confidence": round(float(confidence), 2),
            "target_count": target_count,
            "priority": priority,
            "snapshot_path": snapshot_full_path,
            "metadata": metadata or {}
        }

        # Append to JSON
        existing_data = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []

        existing_data.append(alert_record)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)

        return alert_record
