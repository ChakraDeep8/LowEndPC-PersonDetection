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
        PyTorchDetector
                │
                ├── load_model()
                ├── preprocess()
                ├── inference()
                ├── postprocess()
                ├── draw()
                └── save()
"""

from benchmark.benchmark_runner import BenchmarkRunner
from detectors.pytorch_detector import PyTorchDetector

import config


def main():

    detector = PyTorchDetector()

    runner = BenchmarkRunner(detector)

    runner.warmup(config.INPUT_IMAGE)

    report = runner.benchmark(config.INPUT_IMAGE)

    runner.print_report(report)


if __name__ == "__main__":
    main()