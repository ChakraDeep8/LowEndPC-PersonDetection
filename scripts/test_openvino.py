"""
OpenVINO Model Validation

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import cv2
import numpy as np
import openvino as ov

import config

from pipeline.preprocessor import Preprocessor


def main():

    print("=" * 60)
    print("OpenVINO Model Validation")
    print("=" * 60)

    # --------------------------------------------------
    # OpenVINO Runtime
    # --------------------------------------------------

    print(f"\nOpenVINO Version : {ov.__version__}")

    core = ov.Core()

    # --------------------------------------------------
    # Model Loading
    # --------------------------------------------------

    model_path = config.OPENVINO_MODEL_PATH

    if not model_path.exists():

        raise FileNotFoundError(
            f"OpenVINO model not found: {model_path}"
        )

    print(f"\nLoading Model : {model_path}")

    model = core.read_model(
        model=str(model_path)
    )

    compiled_model = core.compile_model(
        model=model,
        device_name="CPU"
    )

    print("OpenVINO model loaded successfully.")

    # --------------------------------------------------
    # Tensor Information
    # --------------------------------------------------

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print("\nInput Tensor")

    input_names = input_layer.get_names()

    input_name = (
        next(iter(input_names))
        if input_names
        else "input_0"
    )

    print(
        f"  {input_name} : "
        f"{list(input_layer.shape)}"
    )

    print("\nOutput Tensor")

    output_names = output_layer.get_names()

    output_name = (
        next(iter(output_names))
        if output_names
        else "output_0"
    )

    print(
        f"  {output_name} : "
        f"{list(output_layer.shape)}"
    )

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

    results = compiled_model(
        [tensor]
    )

    output = results[output_layer]

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
            "Unexpected OpenVINO output shape. "
            f"Expected {expected_shape}, "
            f"received {output.shape}"
        )

    print("\nOutput shape validation : PASS")

    print("\n" + "=" * 60)
    print("OpenVINO Model Validation Passed")
    print("=" * 60)


if __name__ == "__main__":

    main()