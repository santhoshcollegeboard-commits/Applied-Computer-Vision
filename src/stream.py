"""
Real-time Video Analytics Processing Pipeline.
Integrates EdgeDetector, SpatialEngine, LoiteringTracker, and AlertManager
for multi-rule perimeter monitoring and live throughput telemetry.
"""

import time
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

try:
    from .detector import EdgeDetector
    from .spatial_engine import SpatialEngine
    from .loitering_tracker import LoiteringTracker
    from .alert_manager import AlertManager
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from detector import EdgeDetector
    from spatial_engine import SpatialEngine
    from loitering_tracker import LoiteringTracker
    from alert_manager import AlertManager


class SurveillancePipeline:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        restricted_polygon: List[Tuple[int, int]] = None,
        tripwire_line: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        overcrowding_thresh: int = 5,
        loitering_thresh_sec: float = 6.0,
        alert_cooldown_sec: float = 5.0,
        output_log_path: str = "alerts.json"
    ):
        self.detector = EdgeDetector(model_name=model_name)
        self.spatial = SpatialEngine()
        self.tracker = LoiteringTracker(loiter_threshold_sec=loitering_thresh_sec)
        self.alert_mgr = AlertManager(log_path=output_log_path, cooldown_sec=alert_cooldown_sec)

        # Default restricted zone (polygon) if not provided
        self.restricted_polygon = restricted_polygon or [(150, 100), (500, 100), (500, 400), (150, 400)]
        # Default tripwire (line) if not provided
        self.tripwire_line = tripwire_line or ((0, 250), (640, 250))
        self.overcrowding_thresh = overcrowding_thresh

        self.prev_centroids: Dict[int, Tuple[int, int]] = {}
        self.fps_history: List[float] = []

    def process_frame(self, frame: np.ndarray, current_time: float = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executes end-to-end detection, spatial rule evaluation, loitering tracking, and event alerting.
        Returns annotated frame and dispatched alerts.
        """
        if current_time is None:
            current_time = time.time()

        t0 = time.time()
        detections = self.detector.detect(frame)
        latency_ms = (time.time() - t0) * 1000.0

        annotated = frame.copy()

        # Draw Restricted Zone Polygon (Red boundary)
        pts = np.array(self.restricted_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(annotated, "RESTRICTED ZONE", (self.restricted_polygon[0][0], self.restricted_polygon[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Draw Tripwire Entry Line (Blue line)
        p1, p2 = self.tripwire_line
        cv2.line(annotated, p1, p2, (255, 120, 0), 2)
        cv2.putText(annotated, "TRIPWIRE LINE", (p1[0] + 10, p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)

        centroids = [d['centroid'] for d in detections]
        in_zone_flags = [self.spatial.point_in_polygon(c, self.restricted_polygon) for c in centroids]

        loitering_reports = self.tracker.update(centroids, in_zone_flags, current_time=current_time)
        dispatched_alerts = []

        # Check Overcrowding
        person_count = len(detections)
        if person_count >= self.overcrowding_thresh:
            alert = self.alert_mgr.dispatch_alert(
                event_type="OVERCROWDING_DETECTED",
                frame=annotated,
                confidence=1.0,
                target_count=person_count,
                priority="MEDIUM",
                metadata={"threshold": self.overcrowding_thresh}
            )
            if alert:
                dispatched_alerts.append(alert)
            cv2.putText(annotated, f"ALERT: OVERCROWDING ({person_count})", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check Tripwire Crossing & Zone Intrusions
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d['bbox']
            cx, cy = d['centroid']
            in_zone = in_zone_flags[i]

            # Bounding box color
            box_color = (0, 0, 255) if in_zone else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

            if in_zone:
                alert = self.alert_mgr.dispatch_alert(
                    event_type="RESTRICTED_ZONE_INTRUSION",
                    frame=annotated,
                    confidence=d['confidence'],
                    target_count=person_count,
                    priority="HIGH",
                    metadata={"centroid": (cx, cy)}
                )
                if alert:
                    dispatched_alerts.append(alert)

            # Check Tripwire Crossing
            if cy > p1[1] and (cy - p1[1]) < 35:
                alert = self.alert_mgr.dispatch_alert(
                    event_type="TRIPWIRE_LINE_CROSSING",
                    frame=annotated,
                    confidence=d['confidence'],
                    target_count=person_count,
                    priority="HIGH",
                    metadata={"crossing_y": cy}
                )
                if alert:
                    dispatched_alerts.append(alert)

        # Check Loitering violations
        for track_id, rep in loitering_reports.items():
            tx, ty = rep['centroid']
            dwell = rep['dwell_time']
            cv2.putText(annotated, f"ID {track_id}: {dwell:.1f}s", (tx - 15, ty - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if rep['is_loitering']:
                alert = self.alert_mgr.dispatch_alert(
                    event_type="LOITERING_VIOLATION",
                    frame=annotated,
                    confidence=1.0,
                    target_count=person_count,
                    priority="HIGH",
                    metadata={"track_id": track_id, "dwell_seconds": dwell}
                )
                if alert:
                    dispatched_alerts.append(alert)
                cv2.putText(annotated, f"LOITERING ALERT (ID {track_id})", (tx - 25, ty - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Telemetry overlay
        fps = 1000.0 / max(latency_ms, 1e-3)
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history[-30:])

        cv2.putText(annotated, f"FPS: {avg_fps:.1f} | Latency: {latency_ms:.1f}ms | Targets: {person_count}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return annotated, dispatched_alerts


def run_surveillance_demo(source: str = 'demo', num_frames: int = 30, model_name: str = 'yolov8n.pt', output_path: str = 'examples/surveillance_demo.png'):
    import os
    print("=" * 55)
    print("REAL-TIME EDGE VIDEO SURVEILLANCE PIPELINE")
    print("=" * 55)

    if not os.path.exists(model_name):
        for candidate in ['C:/AI/Working/Project CAM/yolov8n.pt', 'yolov8n.pt', '../yolov8n.pt']:
            if os.path.exists(candidate):
                model_name = candidate
                break

    pipeline = SurveillancePipeline(
        model_name=model_name,
        restricted_polygon=[(50, 50), (250, 50), (250, 250), (50, 250)],
        tripwire_line=((300, 50), (300, 350)),
        loitering_thresh_sec=3.0
    )

    if source == 'demo':
        print(f"Running procedural multi-scenario surveillance stream ({num_frames} frames)...")
        last_frame = None
        total_alerts = 0
        for f_idx in range(num_frames):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 35
            # Draw synthetic pedestrian motion
            px = int(100 + f_idx * 7)
            py = int(150 + np.sin(f_idx * 0.3) * 30)
            cv2.circle(frame, (px, py - 40), 15, (200, 200, 200), -1)
            cv2.rectangle(frame, (px - 18, py - 25), (px + 18, py + 35), (180, 150, 100), -1)

            annotated, alerts = pipeline.process_frame(frame)
            total_alerts += len(alerts)
            last_frame = annotated

        if last_frame is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, last_frame)
            print(f"Surveillance demo snapshot saved to: {output_path}")
        print(f"Completed processing {num_frames} frames. Total alerts dispatched: {total_alerts}")
        return last_frame
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return None
        print(f"Streaming from source: {source} (Press 'q' to exit)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = pipeline.process_frame(frame)
            cv2.imshow("Surveillance Feed", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run real-time edge video surveillance pipeline")
    parser.add_argument('--source', type=str, default='demo', help="Video source: 'demo', '0' for webcam, or path to video file")
    parser.add_argument('--frames', type=int, default=30, help="Number of frames to process in demo mode")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help="Path to YOLOv8 weights (.pt)")
    parser.add_argument('--output', type=str, default='examples/surveillance_demo.png', help="Output path for demo snapshot")
    args = parser.parse_args()

    run_surveillance_demo(source=args.source, num_frames=args.frames, model_name=args.model, output_path=args.output)
