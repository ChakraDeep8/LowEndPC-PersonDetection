"""
System Profiler

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import os
import psutil


class Profiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def snapshot(self):
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.2),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": round(
                self.process.memory_info().rss / (1024 * 1024), 2
            ),
            "threads": self.process.num_threads(),
        }

    def print(self):
        info = self.snapshot()

        print("\nSystem Usage")
        print("-" * 40)
        print(f"CPU Usage      : {info['cpu_percent']} %")
        print(f"RAM Usage      : {info['memory_percent']} %")
        print(f"Process Memory : {info['memory_used_mb']} MB")
        print(f"Threads        : {info['threads']}")