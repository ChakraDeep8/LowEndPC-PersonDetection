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
import config
import json



class BenchmarkLogger:

    def __init__(self):

        self.output_dir = config.CSV
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = config.RESULTS / "metadata.json"

        if not self.metadata_file.exists():

            with open(self.metadata_file, "w", encoding="utf-8") as f:

                json.dump(
                    {
                        "experiment_counter": 0
                    },
                    f,
                    indent=4
                )
        
        today = datetime.now().strftime("%Y-%m-%d")

        self.filename = (
            self.output_dir /
            f"benchmark_{today}.csv"
        )
        
        self.header = [
        "Experiment ID",
        "Run ID",
        "Timestamp",
        "Notes",
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

        if not self.filename.exists():

            with open(self.filename, "w", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)
                writer.writerow(self.header)

    def log(self, notes, row):

        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")

        run_id = self.next_run_id()

        experiment_id = self.next_experiment_id()

        

        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                experiment_id,
                run_id,
                timestamp,
                notes,
                *row
            ])

    def path(self):
        return self.filename
    
    def next_run_id(self):

        with open(self.filename, "r", encoding="utf-8") as f:

            reader = csv.reader(f)

            # Skip header
            next(reader, None)

            return sum(1 for _ in reader) + 1
    
    
    def next_experiment_id(self):

        with open(self.metadata_file, "r", encoding="utf-8") as f:

            metadata = json.load(f)

        metadata["experiment_counter"] += 1

        with open(self.metadata_file, "w", encoding="utf-8") as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        return f"EXP{metadata['experiment_counter']:06d}"               