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

OUTPUT_IMAGES = RESULTS / "images"
CSV = RESULTS / "csv"
LOGS = RESULTS / "logs"


# ==========================
# Input / Output
# ==========================
INPUT_IMAGE = IMAGES / "person.jpeg"
OUTPUT_IMAGES = RESULTS / "images"


# ==========================
# Model
# ==========================

MODEL_NAME = "yolo11n.pt"
MODEL_PATH = MODELS / MODEL_NAME

# ==========================
# Detection
# ==========================

PPERSON_CLASS = 0
CONFIDENCE = 0.25
IOU = 0.45
IMAGE_SIZE = 640
DEVICE = "cpu"


# ==========================
# Benchmark
# ==========================

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10

# ==========================
# Detection
# ==========================

CONFIDENCE = 0.25
IOU = 0.45
IMAGE_SIZE = 640
DEVICE = "cpu"

PERSON_CLASS = 0

# ==========================
# Drawing
# ==========================

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT_SCALE = 0.6

# ==========================
# Save
# ==========================

SAVE_OUTPUT_IMAGE = True
SAVE_CSV = True