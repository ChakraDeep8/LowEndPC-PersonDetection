"""
Project Configuration

LowEndPC-PersonDetection
"""

from pathlib import Path

# ==========================
# Project Directories
# ==========================

ROOT = Path(__file__).parent

MODELS = ROOT / "models"
IMAGES = ROOT / "images"
VIDEOS = ROOT / "videos"

RESULTS = ROOT / "results"
CSV = RESULTS / "csv"
OUTPUT_IMAGES = RESULTS / "images"
LOGS = RESULTS / "logs"

# ==========================
# Model
# ==========================

MODEL_NAME = "yolo11n.pt"
MODEL_PATH = MODELS / MODEL_NAME

# ==========================
# Detection
# ==========================

PERSON_CLASS = 0

CONFIDENCE = 0.15

IOU = 0.45

IMAGE_SIZE = 640

DEVICE = "cpu"

# ==========================
# Benchmark
# ==========================

WARMUP = 3

RUNS = 30