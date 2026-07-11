# Low-End PC Person Detection Benchmark

A modular, backend-independent research framework for benchmarking
**YOLO11 person detection on resource-constrained CPU hardware**.

The project compares inference runtimes under a shared detection
pipeline and measures **latency, throughput, CPU usage, memory
consumption, detection consistency, and deployment behavior**.

> **Current milestone:** PyTorch, ONNX Runtime, OpenVINO, and OpenCV DNN
> are integrated. Cross-backend validation and isolated-process
> benchmarking are implemented.

------------------------------------------------------------------------

## Project Objective

Object detection benchmarks are commonly reported on high-performance
GPUs. Those results do not directly represent older desktops, CPU-only
systems, edge nodes, or low-cost surveillance hardware.

This project investigates:

> **Which inference backend provides the most effective YOLO11 person
> detection deployment on low-end CPU hardware?**

The framework standardizes preprocessing, postprocessing, benchmark
execution, and result logging to make backend comparisons more
consistent and reproducible.

Current reference hardware:

``` text
CPU : Intel Core i3-4150 @ 3.50 GHz
RAM : ~15 GB
Mode: CPU inference
```

------------------------------------------------------------------------

## Key Features

-   Modular `BaseDetector` architecture
-   YOLO11n person detection
-   PyTorch backend
-   ONNX Runtime backend
-   OpenVINO backend
-   OpenCV DNN backend
-   Shared preprocessing pipeline
-   Shared YOLO postprocessing pipeline
-   Person-class filtering and NMS
-   Warm-up and repeated benchmark runs
-   High-resolution timing
-   CPU, RAM, process-memory, and thread profiling
-   CSV experiment logging
-   Persistent experiment IDs
-   Localized timestamps
-   IoU-based backend validation
-   Confidence-difference analysis
-   Isolated-process benchmarking
-   CLI backend selection
-   Webcam and RTSP deployment roadmap

------------------------------------------------------------------------

## Phase Roadmap

### Phase 1: Environment and Architecture

-   [x] Project structure
-   [x] Python environment
-   [x] Dependency management
-   [x] Benchmark modules
-   [x] Detector abstraction
-   [x] Central configuration

### Phase 2: Detection Pipeline

-   [x] `BaseDetector`
-   [x] `PyTorchDetector`
-   [x] Image inference
-   [x] Person-class filtering
-   [x] Detection visualization
-   [x] Shared preprocessing
-   [x] Shared postprocessing
-   [x] Bounding-box scaling
-   [x] Non-Maximum Suppression

### Phase 3: Benchmark Automation

-   [x] High-resolution timing
-   [x] Warm-up runs
-   [x] Repeated benchmark runs
-   [x] CPU and RAM monitoring
-   [x] Process-memory measurement
-   [x] System information collection
-   [x] CSV logging
-   [x] Experiment ID tracking
-   [x] Localized timestamps
-   [x] Backend metadata
-   [x] Success status logging
-   [ ] Performance visualization
-   [ ] Comparative reports

### Phase 4: Multi-Backend Inference

-   [x] PyTorch
-   [x] ONNX Runtime
-   [x] OpenVINO
-   [x] OpenCV DNN
-   [x] Unified YOLO postprocessing
-   [x] Detection matching
-   [x] IoU comparison
-   [x] Confidence-difference comparison
-   [x] ONNX vs OpenVINO validation
-   [x] ONNX vs OpenCV DNN validation
-   [x] Four-backend benchmark

### Phase 5: Isolated and Statistical Benchmarking

-   [x] Independent backend processes
-   [x] Backend-specific lazy imports
-   [x] Isolated process-memory measurement
-   [x] CLI backend selection
-   [ ] Five isolated trials per backend
-   [ ] Mean and median analysis
-   [ ] Standard deviation analysis
-   [ ] Min/max analysis
-   [ ] Statistical backend comparison

### Phase 6: Real-Time Detection

-   [ ] Shared frame-source architecture
-   [ ] Webcam person detection
-   [ ] RTSP person detection
-   [ ] Per-frame inference latency
-   [ ] End-to-end frame latency
-   [ ] Processing and capture FPS
-   [ ] Dropped-frame monitoring
-   [ ] RTSP buffering analysis
-   [ ] Stale-frame latency analysis
-   [ ] Real-time backend comparison

### Phase 7: Optimization

-   [ ] Model quantization
-   [ ] INT8 inference
-   [ ] FP32 vs quantized comparison
-   [ ] Thread-count experiments
-   [ ] CPU affinity experiments
-   [ ] Multi-thread optimization
-   [ ] Backend scalability analysis

### Phase 8: Research Analysis

-   [ ] Performance visualization
-   [ ] Comparative backend reports
-   [ ] Hardware-specific profiles
-   [ ] Statistical interpretation
-   [ ] Optimization trade-off analysis
-   [ ] Final research conclusions

------------------------------------------------------------------------

## Project Structure

``` text
LowEndPC-PersonDetection/
│
├── benchmark/
│   ├── benchmark_runner.py
│   ├── logger.py
│   ├── metrics.py
│   ├── profiler.py
│   ├── system_info.py
│   └── timer.py
│
├── detectors/
│   ├── base_detector.py
│   ├── pytorch_detector.py
│   ├── onnx_detector.py
│   ├── openvino_detector.py
│   └── opencv_detector.py
│
├── pipeline/
│   ├── preprocessor.py
│   └── postprocessor.py
│
├── models/
│   ├── yolo11n.pt
│   ├── yolo11n.onnx
│   └── yolo11n_openvino_model/
│       ├── metadata.yaml
│       ├── yolo11n.bin
│       └── yolo11n.xml
│
├── images/
├── videos/
│
├── results/
│   ├── csv/
│   ├── images/
│   ├── logs/
│   └── metadata.json
│
├── scripts/
│   ├── compare_detections.py
│   ├── export_openvino.py
│   ├── test_openvino.py
│   └── test_opencv.py
│
├── config.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

## Architecture

``` text
                         Input Source
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
                  Image     Webcam     RTSP
                    │       Planned    Planned
                    └─────────┬─────────┘
                              │
                              ▼
                         Preprocessor
                              │
                              ▼
                         BaseDetector
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
 PyTorchDetector        ONNXDetector         OpenVINODetector
                              │                     │
                              └──────────┬──────────┘
                                         │
                                  OpenCVDetector
                                         │
                                         ▼
                                Raw YOLO Prediction
                                         │
                                         ▼
                                    PostProcessor
                                         │
                       Confidence → Person Filter → NMS
                                         │
                                         ▼
                                  Scale Bounding Boxes
                                         │
                                         ▼
                                Normalized Detections
                                         │
                                         ▼
                                  BenchmarkRunner
                                         │
                            Timing → Profiling → Logging
                                         │
                                         ▼
                                      CSV Results
```

------------------------------------------------------------------------

## Supported Backends
``` text
  Backend        Model Format      Device   Status
  -------------- ----------------- -------- -------------
  PyTorch        `.pt`             CPU      Implemented
  ONNX Runtime   `.onnx`           CPU      Implemented
  OpenVINO       `.xml` + `.bin`   CPU      Implemented
  OpenCV DNN     `.onnx`           CPU      Implemented

The optimized runtimes use the shared preprocessing and postprocessing
pipeline to reduce backend-specific processing differences.

------------------------------------------------------------------------
```
## Backend Validation

A backend is not accepted only because the model loads or returns the
expected tensor shape.

The validation pipeline measures:

-   Reference and target detection counts
-   Matched detections
-   Missing detections
-   Extra detections
-   Average, minimum, and maximum IoU
-   Average and maximum confidence difference

Verified on `Crowd.jpeg`:

``` text
ONNX Runtime vs OpenVINO

Reference Detections : 18
Target Detections    : 18
Matched              : 18
Missing              : 0
Extra                : 0

Average IoU          : 1.0000
Minimum IoU          : 1.0000
Maximum IoU          : 1.0000

Average Conf Diff    : 0.0000
Maximum Conf Diff    : 0.0000

Status               : PASS
```

``` text
ONNX Runtime vs OpenCV DNN

Reference Detections : 18
Target Detections    : 18
Matched              : 18
Missing              : 0
Extra                : 0

Average IoU          : 1.0000
Minimum IoU          : 1.0000
Maximum IoU          : 1.0000

Average Conf Diff    : 0.0000
Maximum Conf Diff    : 0.0000

Status               : PASS
```

These results demonstrate detection-level equivalence for the tested
input and current configuration. They do not replace dataset-level
precision, recall, or mAP evaluation against labelled ground truth.

------------------------------------------------------------------------

## Benchmark Methodology

``` text
Input Image
     │
     ▼
Image Read
     │
     ▼
Preprocess
     │
     ▼
Inference
     │
     ▼
Postprocess
     │
     ▼
Draw
     │
     ▼
Save
     │
     ▼
Collect Metrics
     │
     ▼
CSV Logger
```

Current defaults:

``` text
Input Size      : 640
Confidence      : 0.25
IoU Threshold   : 0.45
Device          : CPU
Warm-up Runs    : 3
Benchmark Runs  : 10
```

------------------------------------------------------------------------

## Isolated Backend Benchmarking

Loading several inference runtimes sequentially in one Python process
can contaminate process-memory measurements. Runtime libraries and
allocated memory may remain resident after a backend benchmark
completes.

The project therefore supports isolated backend execution:

``` text
Parent Process
     │
     ├── PyTorch Child Process ──────► Exit
     ├── ONNX Runtime Child Process ─► Exit
     ├── OpenVINO Child Process ─────► Exit
     └── OpenCV DNN Child Process ───► Exit
```

Backend imports are intentionally local. Each child process loads only
the detector runtime required for that experiment.

Run a single backend:

``` bash
python main.py --engine pytorch
python main.py --engine onnx
python main.py --engine openvino
python main.py --engine opencv
```

Run the isolated backend orchestrator:

``` bash
python main.py
```

The next benchmark milestone is **five independent isolated trials per
backend followed by statistical aggregation**.

------------------------------------------------------------------------

## Current Experimental Snapshot

A controlled four-backend run on the reference i3-4150 system produced:

  Backend              Inference   Total Latency         FPS   Detections
  -------------- --------------- --------------- ----------- ------------
  PyTorch             104.408 ms      112.350 ms        8.90           16
  ONNX Runtime         88.733 ms      101.273 ms        9.87           18
  OpenVINO         **74.301 ms**   **87.139 ms**   **11.48**           18
  OpenCV DNN          172.015 ms      184.100 ms        5.43           18

An initial isolated-process run reported:

  Backend          Process Memory
  -------------- ----------------
  PyTorch               811.53 MB
  ONNX Runtime          708.08 MB
  OpenVINO              692.40 MB
  OpenCV DNN        **638.42 MB**

These values are **experimental snapshots, not final backend rankings**.
Significant latency variance has been observed across independent
executions. Formal conclusions will use repeated isolated trials and
statistical aggregation.

------------------------------------------------------------------------

## Logged Benchmark Metrics

The CSV benchmark schema records:

-   Experiment ID
-   Run ID
-   Localized timestamp
-   Experiment notes
-   Engine
-   Backend version
-   Device
-   Model
-   Model format
-   Input
-   Image width and height
-   Model input size
-   CPU
-   Installed RAM
-   Model load time
-   Image read time
-   Preprocess time
-   Inference time
-   Postprocess time
-   Draw time
-   Save time
-   Total latency
-   FPS
-   CPU usage
-   RAM usage
-   Process memory
-   Thread count
-   Detection count
-   Success status

------------------------------------------------------------------------

## Model

Current model:

``` text
YOLO11n
```

Available formats:

``` text
models/yolo11n.pt
models/yolo11n.onnx
models/yolo11n_openvino_model/yolo11n.xml
models/yolo11n_openvino_model/yolo11n.bin
```

The nano variant was selected because this research focuses on
lightweight inference under CPU and memory constraints.

------------------------------------------------------------------------

## Configuration

Core settings are centralized in:

``` text
config.py
```

Configuration includes:

-   Project paths
-   Model paths
-   Input image
-   Confidence threshold
-   IoU threshold
-   Input resolution
-   Device
-   Warm-up runs
-   Benchmark runs
-   Isolated trials
-   Save settings
-   Backend constants
-   Timezone
-   Timestamp format
-   Run mode
-   Input mode
-   Webcam index
-   RTSP configuration

Planned input modes:

``` text
image
webcam
rtsp
```

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone <repository-url>
cd LowEndPC-PersonDetection
```

Create a virtual environment:

``` bash
python -m venv venv
```

Activate on Linux:

``` bash
source venv/bin/activate
```

Activate on Windows:

``` bash
venv\Scripts\activate
```

Upgrade packaging tools:

``` bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

``` bash
python -m pip install -r requirements.txt
```

Verify the environment:

``` bash
python --version
python -m pip check
```

The project currently tracks the environment using `pip freeze`:

``` bash
python -m pip freeze > requirements.txt
```

------------------------------------------------------------------------

## Running the Project

Run the configured benchmark or validation mode:

``` bash
python main.py
```

Run individual isolated backends:

``` bash
python main.py --engine pytorch
python main.py --engine onnx
python main.py --engine openvino
python main.py --engine opencv
```

Validate the OpenVINO model:

``` bash
python -m scripts.test_openvino
```

Validate OpenCV DNN compatibility:

``` bash
python -m scripts.test_opencv
```

Export OpenVINO IR:

``` bash
python -m scripts.export_openvino
```

Compare backend detections:

``` bash
python -m scripts.compare_detections
```

------------------------------------------------------------------------

## Results

``` text
results/
├── csv/
├── images/
├── logs/
└── metadata.json
```

`results/csv/` stores benchmark measurements for statistical analysis.

`results/images/` stores rendered detection outputs.

`results/logs/` is reserved for runtime and benchmark logs.

`results/metadata.json` maintains persistent experiment metadata.

------------------------------------------------------------------------

## Real-Time Detection Roadmap

The image benchmark pipeline will be extended to continuous frame
sources.

### Webcam

Planned measurements:

-   Per-frame inference latency
-   End-to-end frame latency
-   Processing FPS
-   Capture FPS
-   CPU and memory usage
-   Detection stability
-   Failed or dropped frames

### RTSP

Additional RTSP analysis:

-   Network stream capture
-   Buffer growth
-   Stale-frame processing
-   Frame dropping
-   Reconnection behavior
-   Capture-to-inference delay
-   End-to-end stream latency

RTSP experiments will distinguish **model inference latency** from
**stream latency**. High processing FPS alone does not guarantee a
low-latency live pipeline when buffered frames accumulate.

------------------------------------------------------------------------

## Research Questions

1.  Which backend provides the lowest CPU inference latency?
2.  Which backend provides the highest throughput?
3.  How does backend selection affect isolated process memory?
4.  Are optimized runtime detections numerically consistent?
5.  How much performance improvement is achieved through quantization?
6.  Can INT8 materially improve low-end CPU throughput?
7.  How do thread count and CPU affinity affect inference?
8.  Which backend is most suitable for webcam detection?
9.  How does RTSP buffering affect practical real-time latency?
10. Which deployment configuration provides the best
    performance-resource trade-off?

------------------------------------------------------------------------

## Methodology Principles

-   Successful inference does not prove backend equivalence.
-   Matching output shapes do not prove numerical equivalence.
-   Matching tensor min/max values do not prove tensor equivalence.
-   Detection count alone does not prove accuracy.
-   One benchmark execution is insufficient for a final latency ranking.
-   Sequential runtime loading can contaminate process-memory
    measurements.
-   Detection equivalence on one image does not replace dataset-level
    mAP, precision, or recall.
-   Real-time benchmarking must distinguish processing FPS from
    end-to-end latency.

The project therefore combines **backend validation, isolated processes,
repeated trials, and statistical analysis**.

------------------------------------------------------------------------

## Technology Stack

-   Python 3.14
-   PyTorch
-   Ultralytics YOLO11
-   ONNX Runtime
-   OpenVINO
-   OpenCV DNN
-   NumPy
-   Pandas
-   psutil
-   Matplotlib

Current development runtime snapshots include PyTorch `2.13.0+cu130`,
Ultralytics `8.4.92`, ONNX Runtime `1.27.0`, OpenVINO `2026.2.1`, and
OpenCV `5.0.0`.

Exact installed versions are tracked in `requirements.txt`.

------------------------------------------------------------------------

## Project Status

**Active Development**

``` text
Repeated Isolated Benchmarking
              │
              ▼
Statistical Backend Comparison
              │
              ▼
Shared Real-Time Frame Pipeline
              │
              ├── Webcam
              └── RTSP
```

------------------------------------------------------------------------

## License

This project is intended for research and educational purposes.

------------------------------------------------------------------------

## Author

**Deep Chakraborty**

Research interests include computer vision, object detection, image
recognition, deep learning inference optimization, and efficient AI
deployment.
