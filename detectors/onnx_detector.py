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


        
import time
from pipeline.postprocessor import PostProcessor
from pipeline.preprocessor import Preprocessor

class ONNXDetector(BaseDetector):

    def __init__(self):

        super().__init__()

        self.preprocessor = Preprocessor()
        self.postprocessor = PostProcessor()
        self.engine = "ONNX Runtime"
        self.backend_version = ort.__version__
        self.device = "CPU"
        self.model_format = config.ONNX_MODEL_PATH.suffix.lstrip(".")
        self.model_name = config.ONNX_MODEL_PATH.stem
        self.load_model()

    # --------------------------------------------------

    def load_model(self):
        """
        Load ONNX model and measure model loading time.
        """

        import time

        print("Loading ONNX model...")

        start = time.perf_counter()

        self.session = ort.InferenceSession(
            str(config.ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        self.timings["model_load_ms"] = (
            time.perf_counter() - start
        ) * 1000

        print(
            f"ONNX model loaded successfully. "
            f"({self.timings['model_load_ms']:.2f} ms)"
        )

        print("\nInput Tensor")

        for tensor in self.session.get_inputs():

            print(
                f"  {tensor.name} : {tensor.shape}"
            )

        print("\nOutput Tensor")

        for tensor in self.session.get_outputs():

            print(
                f"  {tensor.name} : {tensor.shape}"
            )

        print()


    def preprocess(self, image):
        """
        Preprocess image and retain metadata.
        """

        data = self.preprocessor.process(image)

        self.preprocess_metadata = data

        return data["tensor"]


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

        return self.postprocessor.process(
        predictions,
        self.preprocess_metadata
        )
    
    def detect(self, image_path):
        """
        Complete ONNX detection pipeline.
        """

        # -------------------------
        # Read Image
        # -------------------------

        start = time.perf_counter()

        image = cv2.imread(str(image_path))

        self.timings["image_read_ms"] = (
            time.perf_counter() - start
        ) * 1000

        original = image.copy()

        # -------------------------
        # Preprocess
        # -------------------------

        start = time.perf_counter()

        tensor = self.preprocess(image)

        self.timings["preprocess_ms"] = (
            time.perf_counter() - start
        ) * 1000

        # -------------------------
        # Inference
        # -------------------------

        start = time.perf_counter()

        prediction = self.inference(tensor)

        self.timings["inference_ms"] = (
            time.perf_counter() - start
        ) * 1000

        # -------------------------
        # Postprocess
        # -------------------------

        start = time.perf_counter()

        detections = self.postprocess(prediction)

        self.timings["postprocess_ms"] = (
            time.perf_counter() - start
        ) * 1000

        # -------------------------
        # Draw
        # -------------------------

        output_image = self.draw(
            original,
            detections
        )

        # -------------------------
        # Save
        # -------------------------

        output_path = self.save(
            output_image
        )

        return {
            "persons": len(detections),
            "output": output_path,
            "detections": detections,
            "timings": self.timings
        }


    def draw(self, image, detections):

        """
        Draw person detections on image.
        """

        import cv2
        import time

        start = time.perf_counter()

        for detection in detections:

            x1, y1, x2, y2 = map(
                int,
                detection["bbox"]
            )

            confidence = detection["confidence"]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"Person {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        self.timings["draw_ms"] = (
            time.perf_counter() - start
        ) * 1000

        return image


    def save(self, image):
        """
        Save output image.
        """

        import cv2
        import time

        start = time.perf_counter()

        input_path = config.INPUT_IMAGE

        output_filename = (
            input_path.stem +
            "_result" +
            input_path.suffix
        )

        output_path = config.OUTPUT_IMAGES / output_filename

        cv2.imwrite(
            str(output_path),
            image
        )

        self.timings["save_ms"] = (
            time.perf_counter() - start
        ) * 1000

        return str(output_path)