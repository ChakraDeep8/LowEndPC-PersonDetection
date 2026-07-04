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
from detectors.base_detector import BaseDetector
import config


class PyTorchDetector(BaseDetector):

    def __init__(self):

        super().__init__()

        self.engine = "PyTorch"
        self.model_name = config.MODEL_NAME
        self.model_load_ms = 0.0
        self.image_read_ms = 0.0
        self.load_model()

    # --------------------------------------------------
    # BaseDetector Implementation
    # --------------------------------------------------

    def load_model(self):

        from benchmark.timer import Timer

        print("Loading model...")

        with Timer("Model Loading") as timer:

            self.model = YOLO(str(config.MODEL_PATH))

        self.model_load_ms = timer.elapsed_ms

        print(f"Model Loaded Successfully ({self.model_load_ms:.2f} ms)")

        return self.model

    # --------------------------------------------------

    def preprocess(self, image):

        # Ultralytics handles preprocessing internally.
        return image

    # --------------------------------------------------

    def inference(self, image):

        return self.model.predict(
            source=image,
            imgsz=config.IMAGE_SIZE,
            conf=config.CONFIDENCE,
            iou=config.IOU,
            device=config.DEVICE,
            verbose=False
        )

    # --------------------------------------------------

    def postprocess(self, predictions):

        result = predictions[0]

        detections = []

        for box in result.boxes:

            class_id = int(box.cls)

            if class_id != config.PERSON_CLASS:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])

            detections.append({

                "class_id": class_id,

                "label": "person",

                "confidence": float(box.conf),

                "bbox": [x1, y1, x2, y2]

            })

        return result, detections

    # --------------------------------------------------

    def draw(self, image, detections):

        for detection in detections:

            x1, y1, x2, y2 = map(int, detection["bbox"])

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                config.BOX_COLOR,
                config.BOX_THICKNESS
            )

            cv2.putText(
                image,
                f"{detection['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.BOX_COLOR,
                2
            )

        return image

    # --------------------------------------------------

    def save(self, image, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            image
        )

    # --------------------------------------------------

    def detect(self, image_path):

        image_path = Path(image_path)

        with Timer("Image Read") as timer:

            image = cv2.imread(str(image_path))

        self.image_read_ms = timer.elapsed_ms

        if image is None:
            raise FileNotFoundError(image_path)

        image = self.preprocess(image)

        predictions = self.inference(image)

        result, detections = self.postprocess(predictions)

        annotated = result.plot()

        output_path = (
            config.OUTPUT_IMAGES /
            f"{image_path.stem}_result.jpg"
        )

        self.save(
            annotated,
            output_path
        )

        return {

            "detections": detections,

            "persons": len(detections),

            "output": output_path

        }


if __name__ == "__main__":

    detector = PyTorchDetector()

    result = detector.detect(
        "images/person.jpg"
    )

    print("\nDetection Summary")
    print("-----------------------------")

    print(f"Persons : {result['persons']}")

    print(f"Output  : {result['output']}")