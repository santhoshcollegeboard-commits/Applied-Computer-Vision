"""
Edge Inference Benchmarking and Verification Engine.
Evaluates end-to-end processing latency (ms/frame), throughput (FPS),
spatial rule execution time, and generates benchmark telemetry plots.
"""

import os
import time
import json
import argparse
from typing import Dict, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

try:
    from .stream import SurveillancePipeline
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from stream import SurveillancePipeline


def run_edge_benchmark(
    num_frames: int = 50,
    model_name: str = "yolov8n.pt",
    fig_dir: str = "./examples",
    log_dir: str = "./docs"
) -> Dict[str, Any]:
    print("="*55)
    print("STARTING REAL-TIME EDGE SURVEILLANCE BENCHMARK")
    print("="*55)

    if not os.path.exists(model_name):
        for candidate in ['C:/AI/Working/Project CAM/yolov8n.pt', 'yolov8n.pt', '../yolov8n.pt']:
            if os.path.exists(candidate):
                model_name = candidate
                break

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    pipeline = SurveillancePipeline(
        model_name=model_name,
        output_log_path=os.path.join(log_dir, "alerts.json"),
        loitering_thresh_sec=4.0
    )

    # Generate synthetic video stream of varying surveillance scenarios
    h, w = 480, 640
    latencies = []
    annotated_samples = []

    print(f"Benchmarking {num_frames} surveillance frames on device: {pipeline.detector.device.upper()}...")

    for i in range(num_frames):
        # Create realistic indoor/hallway simulated background
        frame = np.ones((h, w, 3), dtype=np.uint8) * 180
        cv2.line(frame, (0, 320), (w, 320), (140, 140, 140), 2)

        # Simulate moving person entering restricted zone
        sim_time = time.time() + i * 0.2
        target_x = int(120 + i * 8)
        target_y = int(220 + np.sin(i * 0.4) * 20)

        # Render a synthetic person bounding region for detector simulation
        cv2.ellipse(frame, (target_x, target_y), (25, 60), 0, 0, 360, (60, 60, 60), -1)
        cv2.circle(frame, (target_x, target_y - 70), 18, (140, 160, 180), -1)

        t_start = time.time()
        annotated_frame, alerts = pipeline.process_frame(frame, current_time=sim_time)
        latency_ms = (time.time() - t_start) * 1000.0

        latencies.append(latency_ms)

        if i in [10, 25, 45]:
            annotated_samples.append(annotated_frame)

    latencies = np.array(latencies[5:])  # Exclude first 5 frames (warm-up)
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_latency

    metrics = {
        "device": pipeline.detector.device.upper(),
        "model": model_name,
        "total_frames_evaluated": num_frames,
        "mean_latency_ms": round(mean_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "throughput_fps": round(fps, 1),
        "real_time_capable_30fps": bool(fps >= 30.0 or mean_latency < 33.3)
    }

    print("\nBENCHMARK TELEMETRY RESULTS:")
    print(f"Device:              {metrics['device']}")
    print(f"Mean Latency:        {metrics['mean_latency_ms']} ms/frame")
    print(f"95th % Latency:      {metrics['p95_latency_ms']} ms/frame")
    print(f"Throughput:          {metrics['throughput_fps']} FPS")
    print(f"Real-Time Qualified: {metrics['real_time_capable_30fps']}")
    print("="*55)

    # Save latency distribution plot
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(latencies)), latencies, color='#2563eb', lw=2, label='Frame Latency (ms)')
    plt.axhline(mean_latency, color='#dc2626', linestyle='--', label=f'Mean ({mean_latency:.1f} ms)')
    plt.axhline(33.3, color='#059669', linestyle=':', label='30 FPS Threshold (33.3 ms)')
    plt.title('Real-Time Edge Video Inference Latency Profile', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Frame Number', fontsize=11)
    plt.ylabel('Processing Latency (ms)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    chart_path = os.path.join(fig_dir, "latency_benchmark.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency benchmark plot saved to: {chart_path}")

    # Save visual demo sample
    if len(annotated_samples) > 0:
        demo_path = os.path.join(fig_dir, "surveillance_demo.png")
        cv2.imwrite(demo_path, annotated_samples[0])
        print(f"Surveillance demo snapshot saved to: {demo_path}")

    out_json = os.path.join(log_dir, "edge_benchmark_metrics.json")
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {out_json}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    args = parser.parse_args()

    run_edge_benchmark(num_frames=args.frames, model_name=args.model)
