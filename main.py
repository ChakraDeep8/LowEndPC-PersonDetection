"""
LowEndPC-PersonDetection

Main Entry Point

main.py
        │
        ▼
Isolated Backend Runner
        │
        ├── PyTorch Process
        ├── ONNX Runtime Process
        ├── OpenVINO Process
        └── OpenCV DNN Process
                │
                ▼
        BenchmarkRunner
                │
                ├── Timer
                ├── Logger
                ├── Profiler
                ├── Statistics
                └── SystemInfo
"""

import argparse
import subprocess
import sys

from benchmark.benchmark_runner import BenchmarkRunner

import config


# --------------------------------------------------


def create_detector(engine):
    """
    Create detector for the selected backend.

    Local imports keep backend runtimes isolated
    inside their respective benchmark processes.
    """

    if engine == config.PYTORCH:

        from detectors.pytorch_detector import PyTorchDetector

        return PyTorchDetector()

    if engine == config.ONNX:

        from detectors.onnx_detector import ONNXDetector

        return ONNXDetector()

    if engine == config.OPENVINO:

        from detectors.openvino_detector import OpenVINODetector

        return OpenVINODetector()

    if engine == config.OPENCV:

        from detectors.opencv_detector import OpenCVDetector

        return OpenCVDetector()

    raise ValueError(
        f"Unsupported engine: {engine}"
    )


# --------------------------------------------------


def get_benchmark_notes(engine):
    """
    Return benchmark notes for the selected backend.
    """

    notes = {

        config.PYTORCH:
            "PyTorch FP32 Isolated Baseline",

        config.ONNX:
            "ONNX Runtime FP32 Isolated Baseline",

        config.OPENVINO:
            "OpenVINO FP32 Isolated Baseline",

        config.OPENCV:
            "OpenCV DNN FP32 Isolated Baseline"

    }

    return notes[engine]


# --------------------------------------------------


def run_single_benchmark(engine):
    """
    Run one backend benchmark in the current process.
    """

    print("\n" + "=" * 60)

    print(
        f"Isolated Benchmark Backend : {engine}"
    )

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
        notes=get_benchmark_notes(engine)
    )

    runner.print_report(
        report
    )


# --------------------------------------------------


def run_isolated_benchmarks():
    """
    Run every backend in an independent Python process.
    """

    backends = [

        config.PYTORCH,
        config.ONNX,
        config.OPENVINO,
        config.OPENCV

    ]

    print("\n" + "=" * 60)
    print("Isolated Backend Benchmark")
    print("=" * 60)

    for engine in backends:

        print(
            f"\nStarting isolated process: {engine}"
        )

        print("-" * 60)

        command = [

            sys.executable,

            str(
                config.ROOT /
                "main.py"
            ),

            "--engine",

            engine

        ]

        result = subprocess.run(
            command,
            cwd=str(config.ROOT)
        )

        if result.returncode != 0:

            raise RuntimeError(
                "Backend benchmark failed: "
                f"{engine}"
            )

        print(
            f"\nCompleted isolated process: {engine}"
        )

    print("\n" + "=" * 60)

    print(
        "All Isolated Benchmarks Completed"
    )

    print("=" * 60)


# --------------------------------------------------


def run_validation():
    """
    Validate backend numerical equivalence.
    """

    from detectors.onnx_detector import ONNXDetector
    from detectors.opencv_detector import OpenCVDetector

    print(
        "\nRunning Backend Validation...\n"
    )

    reference_detector = ONNXDetector()

    target_detector = OpenCVDetector()

    runner = BenchmarkRunner(
        reference_detector
    )

    runner.validate(

        reference_detector=reference_detector,

        target_detector=target_detector,

        image_path=config.INPUT_IMAGE

    )


# --------------------------------------------------


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "LowEndPC Person Detection Benchmark"
        )
    )

    parser.add_argument(

        "--engine",

        choices=[

            config.PYTORCH,
            config.ONNX,
            config.OPENVINO,
            config.OPENCV

        ],

        default=None,

        help=(
            "Run one backend benchmark "
            "in the current process."
        )

    )

    return parser.parse_args()


# --------------------------------------------------


def main():

    args = parse_arguments()

    # --------------------------------------------------
    # Isolated Child Process
    # --------------------------------------------------

    if args.engine is not None:

        run_single_benchmark(
            args.engine
        )

        return

    # --------------------------------------------------
    # Main Run Mode
    # --------------------------------------------------

    if config.RUN_MODE == config.BENCHMARK:

        run_isolated_benchmarks()

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