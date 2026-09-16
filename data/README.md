# Video Benchmarks & Surveillance Streams Guide

## Overview
This repository provides automated spatial analytics and loitering evaluation for real-time edge CCTV surveillance streams.

## Supported Stream Modalities
1. **Live USB Webcams / Integrated Cameras**: Device indices (`cv2.VideoCapture(0)`).
2. **IP Cameras / RTSP Video Streams**: Network streams (`rtsp://user:pass@camera_ip:554/stream1`).
3. **Pre-recorded Video Files**: Standard video formats (`.mp4`, `.avi`, `.mkv`).
4. **Synthetic Headless Benchmarking**: Integrated frame generator for automated edge latency and throughput testing.

## Public Video Benchmark Reference Datasets
For standardized academic benchmarking, the algorithms in this repository can be evaluated against:
- **PETS 2009 Benchmark** (Performance Evaluation of Tracking and Surveillance): Multi-camera pedestrian tracking, perimeter intrusion, and crowd density sequences.
- **VIRAT Video Dataset**: High-resolution surveillance video for complex human activity and loitering recognition.
- **MOT16 / MOT20** (Multiple Object Tracking Benchmark): Pedestrian detection and trajectory association.

## Model Weights
The detector automatically loads or downloads lightweight edge weights from Ultralytics:
- `yolov8n.pt` (Nano: 3.2M parameters - optimized for high-FPS edge CPU / embedded GPU deployment).
- Weights are cached locally and excluded from Git commits via `.gitignore`.
