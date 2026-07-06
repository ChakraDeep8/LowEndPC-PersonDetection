"""
ONNX YOLO11 Person Detector

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from matplotlib import image
import onnxruntime as ort

from detectors.base_detector import BaseDetector

import config

import numpy as np
import cv2

from pipeline.postprocessor import PostProcessor
from pipeline.preprocessor import Preprocessor

class ONNXDetector(BaseDetector):

    def __init__(self):

        super().__init__()

        self.preprocessor = Preprocessor()
        self.postprocessor = PostProcessor()
        self.engine = "ONNX Runtime"
        self.model_name = config.ONNX_MODEL_NAME
        self.load_model()

    # --------------------------------------------------

    def load_model(self):

        print("Loading ONNX model...")

        self.session = ort.InferenceSession(
            str(config.ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        print("ONNX model loaded successfully.")

        print("\nInput Tensor")

        for tensor in self.session.get_inputs():

            print(
                f"  {tensor.name} : {tensor.shape}"
            )

        print("\nOutput Tensor")

        for tensor in self.session.get_outputs():

            print(
                f"  {tensor.name} : {tensor.shape}\n"
            )

    def preprocess(self, image):

        return self.preprocessor.process(image)


    def inference(self, image):

        input_name = self.session.get_inputs()[0].name

        outputs = self.session.run(
            None,
            {
                input_name: image
            }
        )

        return outputs[0]


    def postprocess(self, predictions):

        return self.postprocessor.process(predictions)

    def draw(self, image, detections):

        raise NotImplementedError(
            "Drawing not implemented yet."
        )


    def save(self, image, output_path):

        raise NotImplementedError(
            "Saving not implemented yet."
        )