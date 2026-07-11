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

INPUT_IMAGE = IMAGES / "Crowd.jpeg"


# ==========================
# Models
# ==========================

MODEL_NAME = "yolo11n.pt"
MODEL_PATH = MODELS / MODEL_NAME

ONNX_MODEL_NAME = "yolo11n.onnx"
ONNX_MODEL_PATH = MODELS / ONNX_MODEL_NAME

OPENVINO_MODEL_DIR = MODELS / "yolo11n_openvino_model" 
OPENVINO_MODEL_NAME = "yolo11n.xml"
OPENVINO_MODEL_PATH = (
    OPENVINO_MODEL_DIR /
    OPENVINO_MODEL_NAME
)


# ==========================
# Detection
# ==========================

PERSON_CLASS = 0

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


# ==========================
# Engines
# ==========================

PYTORCH = "pytorch"
ONNX = "onnx"
OPENVINO = "openvino"
OPENCV = "opencv"

ENGINE = OPENVINO

# ==========================
# Time
# ==========================

TIMEZONE = "Asia/Kolkata"

TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"


# ==========================
# Run Mode
# ==========================

BENCHMARK = "benchmark"
VALIDATION = "validation"

RUN_MODE = BENCHMARK