"""
Spatial Geometry Engine for perimeter security and video analytics.
Implements vector cross-product line crossing (virtual tripwire)
and 2D point-in-polygon testing for restricted perimeter monitoring.
"""

from typing import Tuple, List
import numpy as np


class SpatialEngine:
    """
    Vector geometry engine for real-time spatial surveillance rules.
    """
    @staticmethod
    def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """
        Ray-casting algorithm to determine if 2D coordinate (x, y) is inside a polygonal zone.
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def check_line_crossing(
        prev_point: Tuple[int, int],
        curr_point: Tuple[int, int],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> bool:
        """
        Determines if a moving centroid trajectory (prev_point -> curr_point)
        intersects a virtual tripwire line segment (line_start -> line_end).
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = prev_point, curr_point
        C, D = line_start, line_end

        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
