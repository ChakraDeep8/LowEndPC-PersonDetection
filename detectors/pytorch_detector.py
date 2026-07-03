"""
PyTorch YOLO11 Person Detector

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

from benchmark.timer import Timer
from benchmark.profiler import Profiler
from benchmark.logger import BenchmarkLogger
from benchmark.system_info import SystemInfo
import config


class PyTorchDetector:

    def __init__(self):

        self.logger = BenchmarkLogger()
        self.profiler = Profiler()

        print("Loading model...")

        with Timer("Model Loading") as timer:
            self.model = YOLO(config.MODEL_PATH)

        print(timer)

        print("\nSystem Information")
        SystemInfo.print()

    def detect(self, image_path):

        image_path = Path(image_path)

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        # --------------------------
        # Inference
        # --------------------------

        with Timer("Inference") as infer_timer:

            results = self.model.predict(
                source=image,
                imgsz=config.IMAGE_SIZE,
                conf=config.CONFIDENCE,
                iou=config.IOU,
                device=config.DEVICE,
                verbose=False
            )

        result = results[0]

        persons = 0

        for box in result.boxes:

            cls = int(box.cls)

            if cls != config.PERSON_CLASS:
                continue

            persons += 1

        annotated = result.plot()

        output_path = (
            config.OUTPUT_IMAGES /
            f"{image_path.stem}_result.jpg"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), annotated)

        profile = self.profiler.snapshot()

        fps = 1000 / infer_timer.elapsed_ms

        info = SystemInfo.collect()

        self.logger.log([
            "PyTorch",
            config.MODEL_NAME,
            image_path.name,
            info["cpu"],
            info["ram_gb"],
            0,
            0,
            round(infer_timer.elapsed_ms, 3),
            0,
            0,
            round(infer_timer.elapsed_ms, 3),
            round(fps, 2),
            profile["cpu_percent"],
            profile["memory_percent"],
            profile["memory_used_mb"],
            profile["threads"],
            persons
        ])

        print("\nDetection Complete")
        print("-------------------------------")
        print(f"Persons        : {persons}")
        print(f"Inference Time : {infer_timer.elapsed_ms:.2f} ms")
        print(f"FPS            : {fps:.2f}")
        print(f"Output Image   : {output_path}")
        print(f"CSV            : {self.logger.path()}")


if __name__ == "__main__":

    detector = PyTorchDetector()

    detector.detect(
        "images/person.jpg"
    )