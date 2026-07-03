from benchmark.logger import BenchmarkLogger
from datetime import datetime

logger = BenchmarkLogger()

logger.log([
    datetime.now(),
    "PyTorch",
    "YOLO11n",
    "person.jpg",
    "Intel i3-4150",
    15.02,
    120,
    15,
    48,
    8,
    6,
    197,
    5.08,
    31,
    28,
    215,
    4,
    3
])

print("Saved to")
print(logger.path())