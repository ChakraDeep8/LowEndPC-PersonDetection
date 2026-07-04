"""
Benchmark Logger

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from pathlib import Path
import csv
from datetime import datetime, timezone



class BenchmarkLogger:

    def __init__(self):

        self.output_dir = Path("results/csv")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _timestamp = datetime.now().isoformat(timespec="milliseconds")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.filename = (
            self.output_dir /
            f"benchmark_{timestamp}.csv"
        )
        
        self.header = [
        "Timestamp",
        "Engine",
        "Model",
        "Input",
        "CPU",
        "RAM(GB)",

        "Model Load(ms)",
        "Image Read(ms)",
        "Preprocess(ms)",
        "Inference(ms)",
        "Postprocess(ms)",
        "Draw(ms)",
        "Save(ms)",

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
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")

        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, *row])

    def path(self):
        return self.filename