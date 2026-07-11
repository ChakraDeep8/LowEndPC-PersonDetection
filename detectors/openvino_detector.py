"""
OpenVINO YOLO11 Person Detector

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import time

import cv2
import openvino as ov

from detectors.base_detector import BaseDetector
from pipeline.postprocessor import PostProcessor
from pipeline.preprocessor import Preprocessor

import config


class OpenVINODetector(BaseDetector):

    def __init__(self):

        super().__init__()

        self.preprocessor = Preprocessor()
        self.postprocessor = PostProcessor()

        self.engine = "OpenVINO"
        self.backend_version = ov.__version__
        self.device = "OpenVINO CPU"

        self.model_format = (
            config.OPENVINO_MODEL_PATH
            .suffix
            .lstrip(".")
        )

        self.model_name = (
            config.OPENVINO_MODEL_PATH
            .stem
        )

        self.preprocess_metadata = None

        self.load_model()

    # --------------------------------------------------

    def load_model(self):
        """
        Load OpenVINO model and measure model loading time.
        """

        print("Loading OpenVINO model...")

        if not config.OPENVINO_MODEL_PATH.exists():

            raise FileNotFoundError(
                "OpenVINO model not found: "
                f"{config.OPENVINO_MODEL_PATH}"
            )

        start = time.perf_counter()

        self.core = ov.Core()

        self.model = self.core.read_model(
            model=str(
                config.OPENVINO_MODEL_PATH
            )
        )

        self.compiled_model = self.core.compile_model(
            model=self.model,
            device_name="CPU"
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.timings["model_load_ms"] = (
            time.perf_counter() - start
        ) * 1000

        print(
            "OpenVINO model loaded successfully. "
            f"({self.timings['model_load_ms']:.2f} ms)"
        )

        # -------------------------
        # Input Tensor
        # -------------------------

        input_names = self.input_layer.get_names()

        input_name = (
            next(iter(input_names))
            if input_names
            else "input_0"
        )

        print("\nInput Tensor")

        print(
            f"  {input_name} : "
            f"{list(self.input_layer.shape)}"
        )

        # -------------------------
        # Output Tensor
        # -------------------------

        output_names = self.output_layer.get_names()

        output_name = (
            next(iter(output_names))
            if output_names
            else "output_0"
        )

        print("\nOutput Tensor")

        print(
            f"  {output_name} : "
            f"{list(self.output_layer.shape)}"
        )

        print()

    # --------------------------------------------------

    def preprocess(self, image):
        """
        Preprocess image and retain metadata.
        """

        data = self.preprocessor.process(image)

        self.preprocess_metadata = data

        return data["tensor"]

    # --------------------------------------------------

    def inference(self, image):
        """
        Run OpenVINO inference.
        """

        results = self.compiled_model(
            [image]
        )

        return results[self.output_layer]

    # --------------------------------------------------

    def postprocess(self, predictions):
        """
        Process raw YOLO predictions.
        """

        return self.postprocessor.process(
            predictions,
            self.preprocess_metadata
        )

    # --------------------------------------------------

    def detect(self, image_path):
        """
        Complete OpenVINO detection pipeline.
        """

        # -------------------------
        # Read Image
        # -------------------------

        start = time.perf_counter()

        image = cv2.imread(
            str(image_path)
        )

        self.timings["image_read_ms"] = (
            time.perf_counter() - start
        ) * 1000

        if image is None:

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

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

        detections = self.postprocess(
            prediction
        )

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
            output_image,
            image_path
        )

        return {
            "persons": len(detections),
            "output": output_path,
            "detections": detections,
            "timings": self.timings
        }

    # --------------------------------------------------

    def draw(self, image, detections):
        """
        Draw person detections on image.
        """

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
                config.BOX_COLOR,
                config.BOX_THICKNESS
            )

            cv2.putText(
                image,
                f"Person {confidence:.2f}",
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.BOX_COLOR,
                2
            )

        self.timings["draw_ms"] = (
            time.perf_counter() - start
        ) * 1000

        return image

    # --------------------------------------------------

    def save(self, image, image_path):
        """
        Save output image.
        """

        start = time.perf_counter()

        image_path = config.ROOT / image_path

        output_filename = (
            image_path.stem
            + "_result"
            + image_path.suffix
        )

        output_path = (
            config.OUTPUT_IMAGES
            / output_filename
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        self.timings["save_ms"] = (
            time.perf_counter() - start
        ) * 1000

        return str(output_path)