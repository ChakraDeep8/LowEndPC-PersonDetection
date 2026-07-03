"""
Benchmark Logger

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from pathlib import Path
import csv
from datetime import datetime


class BenchmarkLogger:

    def __init__(self):

        self.output_dir = Path("results/csv")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.filename = (
            self.output_dir /
            f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )

        self.header = [
            "Timestamp",
            "Engine",
            "Model",
            "Input",
            "CPU",
            "RAM(GB)",
            "Load(ms)",
            "Preprocess(ms)",
            "Inference(ms)",
            "Postprocess(ms)",
            "Draw(ms)",
            "Total(ms)",
            "FPS",
            "CPU Usage(%)",
            "RAM Usage(%)",
            "Process Memory(MB)",
            "Threads",
            "Persons"
        ]

        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def log(self, row):

        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def path(self):
        return self.filename