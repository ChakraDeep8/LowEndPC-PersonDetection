"""
LowEndPC-PersonDetection

Main Entry Point

main.py
        │
        ▼
BenchmarkRunner
        │
        ├── Timer
        ├── Logger
        ├── Profiler
        ├── Statistics
        └── SystemInfo
                │
                ▼
        Detector Backend
                │
                ├── PyTorch
                ├── ONNX Runtime
                └── OpenVINO
"""

from benchmark.benchmark_runner import BenchmarkRunner

from detectors.pytorch_detector import PyTorchDetector
from detectors.onnx_detector import ONNXDetector
from detectors.openvino_detector import OpenVINODetector
from detectors.opencv_detector import OpenCVDetector

import config


def create_detector(engine):
    """
    Create detector for the selected backend.
    """

    if engine == config.PYTORCH:

        return PyTorchDetector()

    if engine == config.ONNX:

        return ONNXDetector()

    if engine == config.OPENVINO:

        return OpenVINODetector()
    
    if engine == config.OPENCV:

        return OpenCVDetector()

    raise ValueError(
        f"Unsupported engine: {engine}"
    )


# --------------------------------------------------


def run_benchmark():
    """
    Run controlled benchmark across all backends.
    """

    backends = [

        (
            config.PYTORCH,
            "PyTorch FP32 Baseline"
        ),

        (
            config.ONNX,
            "ONNX Runtime FP32 Baseline"
        ),

        (
            config.OPENVINO,
            "OpenVINO FP32 Baseline"
        )
        ,

        (
            config.OPENCV,
            "OpenCV DNN FP32 Baseline"
        )
    ]

    for engine, notes in backends:

        print("\n" + "=" * 60)
        print(f"Benchmark Backend : {engine}")
        print("=" * 60 + "\n")

        detector = create_detector(
            engine
        )

        runner = BenchmarkRunner(
            detector
        )

        runner.warmup(
            config.INPUT_IMAGE
        )

        report = runner.benchmark(
            config.INPUT_IMAGE,
            notes=notes
        )

        runner.print_report(
            report
        )


# --------------------------------------------------


def run_validation():
    """
    Validate backend numerical equivalence.
    """

    print(
        "\nRunning Backend Validation...\n"
    )

    runner = BenchmarkRunner(
        ONNXDetector()
    )

    runner.validate(

    reference_detector=ONNXDetector(),

    target_detector=OpenCVDetector(),

    image_path=config.INPUT_IMAGE

)

#     runner.validate(

#         reference_detector=ONNXDetector(),

#         target_detector=OpenVINODetector(),

#         image_path=config.INPUT_IMAGE

#     )


        


# --------------------------------------------------


def main():

    if config.RUN_MODE == config.BENCHMARK:

        run_benchmark()

    elif config.RUN_MODE == config.VALIDATION:

        run_validation()
        

    else:

        raise ValueError(
            f"Unsupported run mode: "
            f"{config.RUN_MODE}"
        )


# --------------------------------------------------


if __name__ == "__main__":

    main()