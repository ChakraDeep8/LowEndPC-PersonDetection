"""
Base Detector

Project:
LowEndPC-PersonDetection

Shared by all detector backends. Because every detector measures exactly the same stages.

Therefore:

BaseDetector
        │
        ├── timings
        ├── engine
        └── model_name

becomes shared state.

PyTorch, ONNX, OpenVINO and OpenCV DNN all inherit it.

Author:
Deep Chakraborty
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDetector(ABC):

    def __init__(self):
        self.model = None
        self.engine = ""
        self.model_name = ""
        self.timings = {

        "model_load_ms": 0.0,

        "image_read_ms": 0.0,

        "preprocess_ms": 0.0,

        "inference_ms": 0.0,

        "postprocess_ms": 0.0,

        "draw_ms": 0.0,

        "save_ms": 0.0

    }

    @abstractmethod
    def load_model(self):
        """Load model into memory."""
        pass

    @abstractmethod
    def preprocess(self, image):
        """Prepare input for inference."""
        pass

    @abstractmethod
    def inference(self, image):
        """Run neural network inference."""
        pass

    @abstractmethod
    def postprocess(self, predictions):
        """Convert raw predictions into detections."""
        pass

    @abstractmethod
    def draw(self, image, detections):
        """Draw bounding boxes."""
        pass

    @abstractmethod
    def save(self, image, output_path):
        """Save output image."""
        pass