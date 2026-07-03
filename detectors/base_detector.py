"""
Base Detector

Project:
LowEndPC-PersonDetection

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