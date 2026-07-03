Phase roadmap

Phase 1 
✅ Environment setup

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
│   ├── timer.py
│   ├── profiler.py
│   ├── metrics.py
│   ├── logger.py
│   └── system_info.py
│
├── detectors/
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
│   └── run_benchmark.py
│
├── requirements.txt
├── README.md
└── .gitignore