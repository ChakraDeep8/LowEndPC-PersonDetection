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

import config
from benchmark.timer import Timer
from detectors.base_detector import BaseDetector


class PyTorchDetector(BaseDetector):

    def __init__(self):

        super().__init__()

        self.engine = "PyTorch"
        self.model_name = config.MODEL_NAME

        self.load_model()

    # --------------------------------------------------
    # BaseDetector Implementation
    # --------------------------------------------------

    def load_model(self):
        """
        Load PyTorch YOLO model.
        """

        print("Loading model...")

        with Timer("Model Loading") as timer:

            self.model = YOLO(
                str(config.MODEL_PATH)
            )

        self.timings["model_load_ms"] = timer.elapsed_ms

        print(
            f"Model Loaded Successfully "
            f"({self.timings['model_load_ms']:.2f} ms)"
        )

        return self.model

    # --------------------------------------------------

    def preprocess(
        self,
        image
    ):
        """
        Ultralytics handles preprocessing internally.
        """

        return image

    # --------------------------------------------------

    def inference(
        self,
        image
    ):
        """
        Run PyTorch inference.
        """

        with Timer("Inference") as timer:

            predictions = self.model.predict(
                source=image,
                imgsz=config.IMAGE_SIZE,
                conf=config.CONFIDENCE,
                iou=config.IOU,
                device=config.DEVICE,
                verbose=False
            )

        self.timings["inference_ms"] = timer.elapsed_ms

        return predictions

    # --------------------------------------------------

    def postprocess(
        self,
        predictions
    ):
        """
        Convert Ultralytics results into the
        framework-standard detection format.
        """

        with Timer("Postprocess") as timer:

            result = predictions[0]

            detections = []

            for box in result.boxes:

                class_id = int(box.cls)

                if class_id != config.PERSON_CLASS:
                    continue

                x1, y1, x2, y2 = map(
                    float,
                    box.xyxy[0]
                )

                detections.append({

                    "class_id": class_id,

                    "label": "person",

                    "confidence": float(box.conf),

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]

                })

        self.timings["postprocess_ms"] = timer.elapsed_ms

        return detections

    # --------------------------------------------------

    def draw(
        self,
        image,
        detections
    ):
        """
        Draw detections on image.
        """

        with Timer("Draw") as timer:

            for detection in detections:

                x1, y1, x2, y2 = map(
                    int,
                    detection["bbox"]
                )

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
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE,
                    config.BOX_COLOR,
                    2
                )

        self.timings["draw_ms"] = timer.elapsed_ms

        return image

    # --------------------------------------------------

    def save(
        self,
        image,
        output_path
    ):
        """
        Save annotated image.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with Timer("Save") as timer:

            cv2.imwrite(
                str(output_path),
                image
            )

        self.timings["save_ms"] = timer.elapsed_ms

    # --------------------------------------------------

    def detect(
        self,
        image_path
    ):
        """
        Complete PyTorch detection pipeline.
        """

        image_path = Path(image_path)

        # -------------------------
        # Image Read
        # -------------------------

        with Timer("Image Read") as timer:

            image = cv2.imread(
                str(image_path)
            )

        self.timings["image_read_ms"] = timer.elapsed_ms

        if image is None:

            raise FileNotFoundError(image_path)

        # -------------------------
        # Preprocess
        # -------------------------

        with Timer("Preprocess") as timer:

            image = self.preprocess(image)

        self.timings["preprocess_ms"] = timer.elapsed_ms

        # -------------------------
        # Inference
        # -------------------------

        predictions = self.inference(image)

        # -------------------------
        # Postprocess
        # -------------------------

        detections = self.postprocess(
            predictions
        )

        # -------------------------
        # Draw
        # -------------------------

        annotated = self.draw(
            image.copy(),
            detections
        )

        # -------------------------
        # Save
        # -------------------------

        output_path = (
            config.OUTPUT_IMAGES /
            (
                image_path.stem +
                "_result" +
                image_path.suffix
            )
        )

        self.save(
            annotated,
            output_path
        )

        # -------------------------
        # Return
        # -------------------------

        return {

            "detections": detections,

            "persons": len(detections),

            "output": output_path,

            "timings": self.timings.copy()

        }


if __name__ == "__main__":

    detector = PyTorchDetector()

    result = detector.detect(
        config.INPUT_IMAGE
    )

    print("\nDetection Summary")
    print("-----------------------------")

    print(
        f"Persons : {result['persons']}"
    )

    print(
        f"Output  : {result['output']}"
    )