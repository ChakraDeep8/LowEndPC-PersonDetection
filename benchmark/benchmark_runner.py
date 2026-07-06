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
import gc

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

            gc.collect()

            with Timer(f"Run {i+1}") as timer:

                final_result = self.detector.detect(image_path)

            profile = self.profiler.snapshot()

            timings.append(timer.elapsed_ms)

            cpu_usage.append(profile["cpu_percent"])

            ram_usage.append(profile["memory_percent"])

            process_memory.append(profile["memory_used_mb"])

            print(
                f"Run {i+1:02d}: "
                f"{timer.elapsed_ms:7.2f} ms | "
                f"CPU {profile['cpu_percent']:5.1f}% | "
                f"RAM {profile['memory_percent']:5.1f}%"
            )

        report = {

            "engine": self.detector.engine,

            "model": self.detector.model_name,

            "model_load_ms": self.detector.timings["model_load_ms"],

            "image_read_ms": self.detector.timings["image_read_ms"],

            "preprocess_ms": self.detector.timings["preprocess_ms"],

            "inference_ms": self.detector.timings["inference_ms"],
            
            "postprocess_ms": self.detector.timings["postprocess_ms"],

            "draw_ms": self.detector.timings["draw_ms"],
            
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

        # -----------------------------
        # Basic Information
        # -----------------------------

        report["engine"],                 # Engine
        report["model"],                  # Model
        report["image"],                  # Input
        report["cpu_name"],               # CPU
        report["ram_gb"],                 # RAM

        # -----------------------------
        # Timings
        # -----------------------------

        round(report["model_load_ms"], 3),    # Model Load
        round(report["image_read_ms"], 3),    # Image Read
        round(report["preprocess_ms"], 3),    # Preprocess
        round(report["inference_ms"], 3),     # Inference
        round(report["postprocess_ms"], 3),   # Postprocess
        round(report["draw_ms"], 3),          # Draw
        0,                                    # Save
        round(report["average_ms"], 3),       # Total

        # -----------------------------
        # Performance
        # -----------------------------

        round(report["fps"], 2),
        round(report["cpu_percent"], 2),
        round(report["ram_percent"], 2),
        round(report["process_memory"], 2),

        # -----------------------------
        # Detection
        # -----------------------------

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

        print(f"Image Read Time   : {report['image_read_ms']:.2f} ms")

        print(f"Preprocess Time   : {report['preprocess_ms']:.2f} ms")

        print(f"Inference Time    : {report['inference_ms']:.2f} ms")

        print(f"Postprocess Time  : {report['postprocess_ms']:.2f} ms")

        print(f"Draw Time         : {report['draw_ms']:.2f} ms")

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