Phase roadmap

Phase 1 
Environment setup

Phase 2
PyTorch detector
Image inference
Webcam inference

Phase 3
CSV logging
Benchmark automation
Performance visualization

Phase 4
ONNX Runtime
OpenVINO
OpenCV DNN

Phase 5
Quantization
Multi-thread optimization
Research analysis
###########################################################################################

Project Structure
LowEndPC-PersonDetection/
│
├── benchmark/
│   ├── benchmark_runner.py
│   ├── timer.py
│   ├── profiler.py
│   ├── metrics.py
│   ├── logger.py
│   └── system_info.py
│
├── detectors/
│   ├── base_detector.py
│   ├── pytorch_detector.py
│   ├── onnx_detector.py
│   ├── openvino_detector.py
│   └── opencv_dnn_detector.py
│
├── models/
│   └── yolo11n.pt
│
├── images/
├── videos/
│
├── results/
│   ├── csv/
│   ├── images/
│   └── logs/
│
├── scripts/
│   └── benchmark_system.py
│   └── run_benchmark.py
│   └── test_imread.py
│   └── test_logger.py
│   └── test_profiler.py 
│   └── test_system.py 
│   └── test_timer.py
│
├── requirements.txt
├── main.py
├── README.md
└── .gitignore



Detection Implementation

                    BaseDetector
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
PyTorchDetector     ONNXDetector     OpenVINODetector
     │                   │                   │
     └────────────── Raw Prediction ─────────┘
                         │
                         ▼
                  YOLOPostProcessor
                         │
                         ├── Decode
                         ├── Confidence
                         ├── NMS
                         ├── Scale Boxes
                         └── Normalized Detections
                         │
                         ▼
                  BenchmarkRunner


