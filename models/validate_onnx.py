"""
Validate ONNX Model

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import onnx

import config


def validate_model():

    print("Loading ONNX model...")

    model = onnx.load(str(config.ONNX_MODEL_PATH))

    print("Validating model...")

    onnx.checker.check_model(model)

    print("\nValidation Successful")
    print("=" * 60)

    print(f"IR Version : {model.ir_version}")

    print("\nOpset Imports")

    for opset in model.opset_import:

        print(
            f"  Domain : '{opset.domain}'"
        )

        print(
            f"  Version: {opset.version}"
        )

    print("\nInputs")

    for tensor in model.graph.input:

        dims = []

        for dim in tensor.type.tensor_type.shape.dim:

            dims.append(dim.dim_value)

        print(f"  {tensor.name}: {dims}")

    print("\nOutputs")

    for tensor in model.graph.output:

        dims = []

        for dim in tensor.type.tensor_type.shape.dim:

            dims.append(dim.dim_value)

        print(f"  {tensor.name}: {dims}")

    print("=" * 60)


if __name__ == "__main__":

    validate_model()