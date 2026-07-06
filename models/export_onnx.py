"""
Export YOLO11 Model to ONNX

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from ultralytics import YOLO

import config


def export_model():

    print("Loading PyTorch model...")

    model = YOLO(str(config.MODEL_PATH))

    print("Exporting to ONNX...")

    model.export(
        format="onnx",
        imgsz=config.IMAGE_SIZE,
        opset=17,
        simplify=True,
        dynamic=False
    )

    print("\nExport completed.")

    print(f"PyTorch Model : {config.MODEL_PATH}")
    print(f"ONNX Model    : {config.ONNX_MODEL_PATH}")


if __name__ == "__main__":

    export_model()