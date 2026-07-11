"""
OpenCV DNN Model Validation

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import cv2
import numpy as np

import config

from pipeline.preprocessor import Preprocessor


def main():

    print("=" * 60)
    print("OpenCV DNN Model Validation")
    print("=" * 60)

    # --------------------------------------------------
    # OpenCV Runtime
    # --------------------------------------------------

    print(f"\nOpenCV Version : {cv2.__version__}")

    # --------------------------------------------------
    # Model Loading
    # --------------------------------------------------

    model_path = config.ONNX_MODEL_PATH

    if not model_path.exists():

        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    print(f"\nLoading Model : {model_path}")

    network = cv2.dnn.readNetFromONNX(
        str(model_path)
    )

    network.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    network.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    print("OpenCV DNN model loaded successfully.")

    # --------------------------------------------------
    # Image Loading
    # --------------------------------------------------

    image = cv2.imread(
        str(config.INPUT_IMAGE)
    )

    if image is None:

        raise FileNotFoundError(
            f"Image not found: {config.INPUT_IMAGE}"
        )

    # --------------------------------------------------
    # Shared Preprocessing
    # --------------------------------------------------

    preprocessor = Preprocessor()

    data = preprocessor.process(image)

    tensor = data["tensor"]

    print("\nPreprocessed Tensor")

    print(f"  Shape : {tensor.shape}")
    print(f"  Type  : {tensor.dtype}")
    print(f"  Min   : {tensor.min():.6f}")
    print(f"  Max   : {tensor.max():.6f}")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    network.setInput(tensor)

    output = network.forward()

    # --------------------------------------------------
    # Output Validation
    # --------------------------------------------------

    print("\nInference Output")

    print(f"  Shape : {output.shape}")
    print(f"  Type  : {output.dtype}")
    print(f"  Min   : {np.min(output):.6f}")
    print(f"  Max   : {np.max(output):.6f}")

    expected_shape = (
        1,
        84,
        8400
    )

    if output.shape != expected_shape:

        raise ValueError(
            "Unexpected OpenCV DNN output shape. "
            f"Expected {expected_shape}, "
            f"received {output.shape}"
        )

    print("\nOutput shape validation : PASS")

    print("\n" + "=" * 60)
    print("OpenCV DNN Model Validation Passed")
    print("=" * 60)


if __name__ == "__main__":

    main()