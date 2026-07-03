"""
High Precision Timer

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty

Description:
Reusable timer for benchmarking different stages of
the computer vision pipeline.
"""

from time import perf_counter_ns


class Timer:
    def __init__(self, name="Task"):
        self.name = name
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        self.start_time = perf_counter_ns()

    def stop(self):
        self.end_time = perf_counter_ns()

    @property
    def elapsed_ns(self):
        return self.end_time - self.start_time

    @property
    def elapsed_us(self):
        return self.elapsed_ns / 1_000

    @property
    def elapsed_ms(self):
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_sec(self):
        return self.elapsed_ns / 1_000_000_000

    def reset(self):
        self.start_time = 0
        self.end_time = 0

    def summary(self):
        return {
            "task": self.name,
            "seconds": self.elapsed_sec,
            "milliseconds": self.elapsed_ms,
            "microseconds": self.elapsed_us,
            "nanoseconds": self.elapsed_ns,
        }

    def __str__(self):
        return (
            f"{self.name}\n"
            f"  Seconds      : {self.elapsed_sec:.6f}\n"
            f"  Milliseconds : {self.elapsed_ms:.3f}\n"
            f"  Microseconds : {self.elapsed_us:.3f}\n"
            f"  Nanoseconds  : {self.elapsed_ns}"
        )