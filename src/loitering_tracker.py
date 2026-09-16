"""
Temporal Tracking & Loitering Analysis Engine.
Maintains centroid trajectories across video frames, tracks dwell duration
inside monitored zones, and flags prolonged loitering anomalies.
"""

import time
from typing import Dict, List, Tuple
import numpy as np


class LoiteringTracker:
    def __init__(self, max_disappeared: int = 15, loiter_threshold_sec: float = 8.0, dist_threshold: float = 60.0):
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.first_seen: Dict[int, float] = {}
        self.in_restricted_zone: Dict[int, bool] = {}
        self.zone_entry_time: Dict[int, float] = {}

        self.max_disappeared = max_disappeared
        self.loiter_threshold_sec = loiter_threshold_sec
        self.dist_threshold = dist_threshold

    def register(self, centroid: Tuple[int, int], in_zone: bool = False, current_time: float = None):
        if current_time is None:
            current_time = time.time()
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.first_seen[obj_id] = current_time
        self.in_restricted_zone[obj_id] = in_zone
        self.zone_entry_time[obj_id] = current_time if in_zone else None
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.first_seen[obj_id]
        del self.in_restricted_zone[obj_id]
        del self.zone_entry_time[obj_id]

    def update(self, input_centroids: List[Tuple[int, int]], in_zone_flags: List[bool], current_time: float = None):
        """
        Associates input centroids with existing tracks using Euclidean distance.
        """
        if current_time is None:
            current_time = time.time()

        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return {}

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, in_zone=in_zone_flags[i], current_time=current_time)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.dist_threshold:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0

                # Update zone status
                is_in_zone = in_zone_flags[col]
                if is_in_zone and not self.in_restricted_zone[obj_id]:
                    self.zone_entry_time[obj_id] = current_time
                elif not is_in_zone:
                    self.zone_entry_time[obj_id] = None

                self.in_restricted_zone[obj_id] = is_in_zone
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(input_centroids[col], in_zone=in_zone_flags[col], current_time=current_time)

        # Check for loitering violations
        status_reports = {}
        for obj_id, in_zone in self.in_restricted_zone.items():
            if in_zone and self.zone_entry_time[obj_id] is not None:
                dwell_sec = current_time - self.zone_entry_time[obj_id]
                is_loitering = (dwell_sec >= self.loiter_threshold_sec)
                status_reports[obj_id] = {
                    'centroid': self.objects[obj_id],
                    'dwell_time': round(dwell_sec, 1),
                    'is_loitering': is_loitering
                }

        return status_reports
