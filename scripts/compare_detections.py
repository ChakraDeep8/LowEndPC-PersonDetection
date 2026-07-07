"""
Compare detections produced by different backends.

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import cv2

import config

from detectors.pytorch_detector import PyTorchDetector
from detectors.onnx_detector import ONNXDetector
from utils.matcher import match_detections


def main():

    image = cv2.imread(str(config.INPUT_IMAGE))

    pytorch = PyTorchDetector()

    onnx = ONNXDetector()

    pytorch_tensor = pytorch.preprocess(image.copy())
    onnx_tensor = onnx.preprocess(image.copy())

    pytorch_prediction = pytorch.inference(pytorch_tensor)
    onnx_prediction = onnx.inference(onnx_tensor)

    pytorch_detections = pytorch.postprocess(
        pytorch_prediction
    )

    onnx_detections = onnx.postprocess(
        onnx_prediction
    )

    matches = match_detections(
        pytorch_detections,
        onnx_detections
    )

    print("\n" + "=" * 60)
    print("Detection Comparison")
    print("=" * 60)

    print(
        f"PyTorch Detections : {len(pytorch_detections)}"
    )

    print(
        f"ONNX Detections    : {len(onnx_detections)}"
    )

    print(
        f"Matched            : {len(matches)}"
    )

    print()

    for index, match in enumerate(matches, start=1):

        reference = match["reference"]
        target = match["target"]

        print("-" * 60)

        print(f"Detection {index}")

        print()

        print(
            f"IoU        : {match['iou']:.4f}"
        )

        print(
            f"PyTorch    : {reference['bbox']}"
        )

        print(
            f"Confidence : {reference['confidence']:.4f}"
        )

        print()

        print(
            f"ONNX       : {target['bbox']}"
        )

        print(
            f"Confidence : {target['confidence']:.4f}"
        )

        print()

    print("=" * 60)


if __name__ == "__main__":

    main()