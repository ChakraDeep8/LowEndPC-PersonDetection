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
from detectors.onnx_detector import ONNXDetector
from benchmark.system_info import SystemInfo

import config


def main():

        if config.ENGINE == config.PYTORCH:

                detector = PyTorchDetector()

        elif config.ENGINE == config.ONNX:

                detector = ONNXDetector()

        else:

                raise ValueError(
                        f"Unsupported engine: {config.ENGINE}"
                )

        # SystemInfo.save()

        runner = BenchmarkRunner(detector)

        runner.warmup(config.INPUT_IMAGE)

        if config.RUN_MODE == config.BENCHMARK:

                report = runner.benchmark(
                        config.INPUT_IMAGE,
                        notes="PyTorch Baseline"
                )

                runner.print_report(report)

        elif config.RUN_MODE == config.VALIDATION:

                print("\nRunning Backend Validation...\n")

                runner.validate(
                        reference_detector=PyTorchDetector(),
                        target_detector=ONNXDetector(),
                        image_path=config.INPUT_IMAGE
                )

        else:

                raise ValueError(
                        f"Unsupported run mode: {config.RUN_MODE}"
                )

if __name__ == "__main__":
    main()