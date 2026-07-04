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

from detectors.pytorch_detector import PyTorchDetector
from benchmark.benchmark_runner import BenchmarkRunner


def main():

    detector = PyTorchDetector()

    runner = BenchmarkRunner(detector)

    runner.warmup("images/person.jpg")

    report = runner.benchmark("images/person.jpg")

    runner.print_report(report)


if __name__ == "__main__":

    main()