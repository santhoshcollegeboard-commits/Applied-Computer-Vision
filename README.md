# Real-Time Edge Video Surveillance & Spatial-Temporal Analytics

Real-time edge video surveillance pipeline combining YOLOv8 pedestrian detection, vector geometry tripwires, polygonal restricted zoning, loitering tracking, and CPU latency benchmarking.

## Quick Start

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/santhoshcollegeboard-commits/applied-computer-vision.git
cd applied-computer-vision
pip install -r requirements.txt

# 2. Run real-time surveillance demo (synthetic procedural stream)
python src/stream.py --source demo --frames 30

# 3. Run live webcam monitoring (if webcam available)
python src/stream.py --source 0

# 4. Execute edge latency and throughput benchmark
python src/benchmark.py --frames 40
```

## Overview

Traditional perimeter surveillance systems rely heavily on manual video monitoring or primitive pixel-differencing motion detection, which frequently triggers false alarms from lighting shifts, weather phenomena, and wind-blown foliage. Furthermore, modern surveillance demands complex spatial logic—such as directional virtual tripwires, multi-point polygonal restricted zones, and temporal dwell-time loitering tracking—executed locally on edge hardware with strict latency constraints.

This repository provides an end-to-end edge video analytics architecture built in Python, OpenCV, and Ultralytics YOLOv8. The system integrates 2D computational vector geometry (cross-product line crossing) and ray-casting point-in-polygon zoning with an Euclidean centroid tracker to detect perimeter breaches, log structured incident alerts to JSON, and save annotated visual evidence snapshots.

## Problem Statement

Deploying computer vision models on resource-constrained edge computing nodes requires balancing detection accuracy against inference throughput and computational complexity. Monolithic video pipelines frequently suffer from frame dropping, race conditions, and lack of deterministic spatial perimeter enforcement. An effective edge system must maintain real-time telemetry while computing complex spatial-temporal queries per frame.

## What This Project Does

- **Real-Time Target Localization**: Executes lightweight YOLOv8 pedestrian detection and bounding box extraction.
- **Directional Virtual Tripwires**: Implements 2D vector cross-product geometry to detect when a target crosses a defined line segment and determine crossing directionality.
- **Polygonal Restricted Zoning**: Employs ray-casting point-in-polygon algorithms to continuously monitor arbitrary convex and non-convex restricted security perimeters.
- **Temporal Loitering Tracking**: Features an Euclidean centroid tracker that monitors individual target dwell times and escalates loitering alerts upon exceeding a 4.0-second threshold.
- **Structured Incident Logging**: Automatically writes JSON alert telemetry and saves visual snapshot frames to disk with configurable alert cooldown gating.
- **Edge Latency Profiling**: Measures per-frame inference time, 95th percentile latency, and frame-rate throughput (FPS) across edge hardware.

## Main Program

### MAIN ENTRY POINT:
`src/stream.py`
The central video analytics engine that connects video ingestion (webcam, file, or synthetic demo), target detection, spatial geometry evaluation, temporal loitering tracking, telemetry overlays, and alert dispatching.

### TRAINING / MODEL CONFIGURATION:
`src/detector.py`
Configures Ultralytics YOLOv8 nano/small models, class filtering (person class ID 0), confidence thresholds, and hardware dispatch (CPU/CUDA).

### EVALUATION & BENCHMARKING:
`src/benchmark.py`
Command: `python src/benchmark.py --frames 40`
Executes latency profiling across frame sequences, evaluates mean/p95 execution times, and exports latency distribution plots and JSON telemetry.

### INFERENCE / DEMO:
`src/stream.py`
Command: `python src/stream.py --source demo --frames 30`
Executes an interactive procedural surveillance stream demonstrating tripwire crossings, zone intrusions, and loitering tracking, saving a verification snapshot to `examples/surveillance_demo.png`.

## Project Structure

```
applied-computer-vision/
├── .gitignore               # Excludes virtual environments, weights, videos, and logs
├── LICENSE                  # MIT License
├── README.md                # Comprehensive system documentation and benchmark report
├── requirements.txt         # Core dependencies (Ultralytics YOLOv8, PyTorch, OpenCV, NumPy)
├── data/
│   └── README.md            # Video stream source guidance and simulation instructions
├── docs/
│   └── edge_benchmark_metrics.json # Machine-readable empirical latency metrics
├── examples/
│   ├── latency_benchmark.png # Edge latency per-frame profiling chart
│   └── surveillance_demo.png # Annotated surveillance frame with tripwire and zones
└── src/
    ├── __init__.py          # Package initialization
    ├── detector.py          # YOLOv8 object detection and class filtering
    ├── spatial_engine.py    # Vector cross-product tripwires and ray-casting polygon zoning
    ├── loitering_tracker.py # Euclidean centroid tracking and temporal dwell accumulation
    ├── alert_manager.py     # Structured JSON logging and snapshot capture
    ├── stream.py            # Main video analytics pipeline and demo CLI
    └── benchmark.py         # Latency profiling and edge throughput evaluation
```

## Dataset & Video Sources

- **Detection Model**: Pre-trained COCO-initialized weights (`yolov8n.pt`, 6.2 MB), automatically managed via Ultralytics.
- **Video Input Streams**:
  1. `demo`: Built-in procedural multi-scenario synthetic stream generating simulated pedestrians traversing across virtual perimeters.
  2. `0` / Webcams: Direct live camera feed ingestion via OpenCV `VideoCapture`.
  3. Video Files: Compatible with standard MP4, AVI, and MKV video surveillance recordings.

## Methodology

### 1. Spatial Vector Geometry (Tripwires)
Given a virtual tripwire defined by segment $AB$ from $A(x_1, y_1)$ to $B(x_2, y_2)$ and consecutive target positions $P_{t-1}$ and $P_t$:
- The vector orientation is calculated using 2D cross-products:
  $$\text{Cross}(A, B, P) = (B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x)$$
- A crossing occurs if and only if:
  $$\text{Sign}(\text{Cross}(A, B, P_{t-1})) \ne \text{Sign}(\text{Cross}(A, B, P_t)) \quad \text{and} \quad \text{SegmentIntersect}(AB, P_{t-1}P_t) = \text{True}$$

### 2. Polygonal Perimeter Zoning
Evaluates target centroid $(x_c, y_c)$ against arbitrary $N$-sided polygons using ray-casting:
- A horizontal ray is cast from $(x_c, y_c)$ toward positive infinity; the point is inside the restricted zone if the ray intersects the polygon boundary an odd number of times.

### 3. Euclidean Centroid Loitering Tracking
- Target centroids in frame $t$ are matched to active track IDs in frame $t-1$ by minimizing pairwise Euclidean distance.
- Dwell time is accumulated: $\Delta t = t_{\text{current}} - t_{\text{first\_seen}}$.
- When $\Delta t \ge 4.0\text{ seconds}$, a high-priority `LOITERING_VIOLATION` is dispatched.

## Model

### YOLOv8 Nano (`yolov8n.pt`)
- **Parameter Count**: 3.2 Million parameters
- **Architecture**: Modified CSPDarknet backbone with Path Aggregation Network (PAN) neck and anchor-free decoupled detection head.
- **Filtered Classes**: Person (Class ID 0) with confidence threshold $\tau = 0.40$ and Non-Maximum Suppression (NMS) IoU threshold of 0.45.

## Evaluation & Benchmarking

```bash
python src/benchmark.py --frames 40 --model yolov8n.pt
```

Benchmarking records latency per frame encompassing:
1. Frame reading and color conversion
2. Neural forward inference
3. NMS post-processing
4. Centroid association and loitering updates
5. Vector spatial cross-product and polygon checks
6. Visual frame annotation and telemetry rendering

## Results

Empirical latency and throughput measured across 40 continuous surveillance frames on native CPU:

| Benchmark Metric | Measured Result | Operational Characterization |
| :--- | :---: | :--- |
| **Evaluated Hardware** | **Native CPU** | Standard edge host processor |
| **Detection Model** | **YOLOv8n (3.2M params)** | Lightweight edge-optimized network |
| **Total Frames Profiled** | **40 frames** | Continuous multi-scenario sequence |
| **Mean Frame Latency** | **104.94 ms/frame** | Stable ~105 ms end-to-end processing |
| **95th Percentile (p95) Latency** | **147.72 ms/frame** | Upper-bound latency under multi-target load |
| **Minimum Latency** | **46.98 ms/frame** | Baseline latency in sparse frames |
| **Maximum Latency** | **179.73 ms/frame** | Peak latency during alert serialization |
| **CPU Throughput** | **9.5 FPS** | Continuous CPU stream processing |
| **Loitering Escalation Threshold** | **4.0 seconds** | Reliable temporal dwell-time triggering |

## Example Output

- **Latency Telemetry Profile**: Visualized in `examples/latency_benchmark.png`, depicting frame-by-frame execution time alongside moving average trendlines and threshold boundaries.
- **Surveillance Frame Snapshot**: Rendered in `examples/surveillance_demo.png`, illustrating active tripwire lines, green/red restricted zone boundary overlays, individual target track IDs with real-time dwell timers, and top-left FPS/latency diagnostics.

## Limitations

- **Native CPU Real-Time Ceiling**: At 104.9 ms/frame (~9.5 FPS on CPU), native host processing cannot reach full 30 FPS without GPU or hardware accelerator delegation (e.g., TensorRT, OpenVINO, or NPU quantization).
- **Occlusion Tracking**: Centroid distance matching can switch track IDs when two individuals closely overlap for multiple consecutive frames; deep metric ReID embeddings (DeepSORT / ByteTrack) would improve heavy crowding resilience.
- **Monocular Scale Distortion**: Pixel-distance loitering does not account for 3D perspective distortion (distant pedestrians appear to move slower in pixel coordinates than foreground pedestrians).

## How to Run

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Synthetic Demo Surveillance Stream
```bash
python src/stream.py --source demo --frames 30
```

### 3. Run Live Camera Monitoring
```bash
python src/stream.py --source 0
```

### 4. Run Video File Analytics
```bash
python src/stream.py --source path/to/surveillance_video.mp4
```

### 5. Run Latency Benchmark
```bash
python src/benchmark.py --frames 40
```

## Future Work

1. **TensorRT / OpenVINO Optimization**: Export YOLOv8 to FP16/INT8 TensorRT engines to push edge inference throughput from 9.5 FPS to >60 FPS on edge accelerators (Jetson Orin).
2. **ByteTrack Multi-Object Integration**: Replace centroid matching with ByteTrack association using low-score detection recycling to enhance trajectory continuity through occlusions.
3. **Camera Calibration & Perspective Homography**: Calibrate ground plane homography matrix ($3\times 3$) to convert pixel speeds to metric units ($km/h$ and real-world meters).
