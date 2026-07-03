"""
Benchmark Runner

Project:
LowEndPC-PersonDetection
"""
"""
Instead of every detector implementing benchmarking:

PyTorch             ONNX            OpenVINO
    ↓                 ↓                 ↓
Benchmark         Benchmark         Benchmark


I separate them:

                 BenchmarkRunner
                       │
      ┌────────────────┼────────────────┐
      │                │                │
 PyTorchDetector  ONNXDetector  OpenVINODetector

This follows the Strategy Pattern. 
It's easy to plug different detectors into the same benchmarking engine.

"""

import statistics

from benchmark.timer import Timer
from benchmark.profiler import Profiler
from benchmark.logger import BenchmarkLogger
from benchmark.system_info import SystemInfo


class BenchmarkRunner:

    def __init__(self, detector):

        self.detector = detector

        self.logger = BenchmarkLogger()

        self.profiler = Profiler()

        self.system = SystemInfo.collect()

    def warmup(self, image, runs=3):

        print(f"Warmup ({runs} runs)")

        for _ in range(runs):
            self.detector.detect(image)

    def benchmark(self, image, runs=30):

        timings = []

        print(f"Benchmark ({runs} runs)")

        for i in range(runs):

            with Timer() as timer:
                self.detector.detect(image)

            timings.append(timer.elapsed_ms)

            print(
                f"Run {i+1:02d}: "
                f"{timer.elapsed_ms:.2f} ms"
            )

        return self.calculate_statistics(timings)

    def calculate_statistics(self, values):

        return {
            "average": statistics.mean(values),
            "minimum": min(values),
            "maximum": max(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "fps": 1000 / statistics.mean(values)
        }