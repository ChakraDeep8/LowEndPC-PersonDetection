"""
Export YOLO11n to OpenVINO IR

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from pathlib import Path

from ultralytics import YOLO

import config


def main():
    """
    Export the PyTorch YOLO11 model to OpenVINO IR.
    """

    print("=" * 60)
    print("YOLO11 → OpenVINO Export")
    print("=" * 60)

    if not config.MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {config.MODEL_PATH}"
        )

    print(f"\nLoading Model : {config.MODEL_PATH}")

    model = YOLO(str(config.MODEL_PATH))

    print("Model loaded successfully.")

    print("\nExporting to OpenVINO...")

    exported_path = model.export(

        format="openvino",

        imgsz=config.IMAGE_SIZE,

        dynamic=False,

        half=False,

        int8=False,

        simplify=True,

        nms=False
    )

    print("\nExport completed successfully.")

    print(f"\nOutput Directory : {exported_path}")

    output_dir = Path(exported_path)

    print("\nGenerated Files:")

    for file in sorted(output_dir.iterdir()):

        print(f"  {file.name}")

    print("\n" + "=" * 60)
    print("OpenVINO Export Finished")
    print("=" * 60)


if __name__ == "__main__":

    main()