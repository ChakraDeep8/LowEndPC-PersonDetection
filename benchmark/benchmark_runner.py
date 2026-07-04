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

Notice that BenchmarkRunner never imports Ultralytics.

It only understands normalized detections.

This separation makes the framework extensible.
"""

"""
Benchmark Runner

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

"""
Benchmark Runner

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from statistics import mean, median, stdev

from benchmark.timer import Timer
from benchmark.profiler import Profiler
from benchmark.logger import BenchmarkLogger
from benchmark.system_info import SystemInfo

import config


class BenchmarkRunner:

    def __init__(self, detector):

        self.detector = detector

        self.logger = BenchmarkLogger()

        self.profiler = Profiler()

        self.system = SystemInfo.collect()

    def warmup(self, image_path):

        print(f"\nWarmup ({config.WARMUP_RUNS} runs)")

        for i in range(config.WARMUP_RUNS):

            self.detector.detect(image_path)

            print(f"Warmup {i + 1}/{config.WARMUP_RUNS}")

    def benchmark(self, image_path):

        timings = []

        cpu_usage = []

        ram_usage = []

        process_memory = []

        final_result = None

        print(f"\nBenchmark ({config.BENCHMARK_RUNS} runs)\n")

        for i in range(config.BENCHMARK_RUNS):

            with Timer(f"Run {i+1}") as timer:

                final_result = self.detector.detect(image_path)

            profile = self.profiler.snapshot()

            timings.append(timer.elapsed_ms)

            cpu_usage.append(profile["cpu_percent"])

            ram_usage.append(profile["memory_percent"])

            process_memory.append(profile["memory_used_mb"])

            print(
                f"Run {i+1:02d}: "
                f"{timer.elapsed_ms:.2f} ms"
            )

        report = {

            "engine": self.detector.engine,

            "model": self.detector.model_name,

            "model_load_ms": self.detector.model_load_ms,

            "cpu_name": self.system["cpu"],

            "ram_gb": self.system["ram_gb"],

            "image": image_path,

            "average_ms": mean(timings),

            "minimum_ms": min(timings),

            "maximum_ms": max(timings),

            "median_ms": median(timings),

            "std_ms": stdev(timings) if len(timings) > 1 else 0,

            "fps": 1000 / mean(timings),

            "cpu_percent": mean(cpu_usage),

            "ram_percent": mean(ram_usage),

            "process_memory": mean(process_memory),

            "persons": final_result["persons"],

            "output": final_result["output"]

        }

        self.save(report)

        return report

    def save(self, report):

        self.logger.log([

    report["engine"],                 # Engine
    report["model"],                  # Model
    round(report["model_load_ms"], 3),# Model Load
    report["image"],                  # Input
    report["cpu_name"],               # CPU
    report["ram_gb"],                 # RAM

    0,                                # Image Read
    0,                                # Preprocess

    round(report["average_ms"], 3),   # Inference

    0,                                # Postprocess
    0,                                # Draw
    0,                                # Save

    round(report["average_ms"], 3),   # Total

    round(report["fps"], 2),

    round(report["cpu_percent"], 2),

    round(report["ram_percent"], 2),

    round(report["process_memory"], 2),

    self.profiler.process.num_threads(),

    report["persons"]

])

    def print_report(self, report):

        print()

        print("=" * 60)

        print("Benchmark Report")

        print("=" * 60)

        print(f"Engine            : {report['engine']}")

        print(f"Model             : {report['model']}")

        print(f"Model Load Time   : {report['model_load_ms']:.2f} ms")

        print(f"Average Latency   : {report['average_ms']:.2f} ms")

        print(f"Minimum Latency   : {report['minimum_ms']:.2f} ms")

        print(f"Maximum Latency   : {report['maximum_ms']:.2f} ms")

        print(f"Median Latency    : {report['median_ms']:.2f} ms")

        print(f"Std Deviation     : {report['std_ms']:.2f} ms")

        print(f"Average FPS       : {report['fps']:.2f}")

        print(f"CPU Usage         : {report['cpu_percent']:.2f} %")

        print(f"RAM Usage         : {report['ram_percent']:.2f} %")

        print(f"Process Memory    : {report['process_memory']:.2f} MB")

        print(f"Persons Detected  : {report['persons']}")

        print(f"Output Image      : {report['output']}")

        print("=" * 60)