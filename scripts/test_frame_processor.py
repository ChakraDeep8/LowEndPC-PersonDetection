"""
Shared Frame Processor Validation

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import config

from pipeline.frame_processor import FrameProcessor
from pipeline.frame_source import FrameSource


BACKENDS = [
    config.PYTORCH,
    config.ONNX,
    config.OPENVINO,
    config.OPENCV,
]


def create_detector(engine):
    """
    Create detector using local backend imports.
    """

    if engine == config.PYTORCH:

        from detectors.pytorch_detector import PyTorchDetector

        return PyTorchDetector()

    if engine == config.ONNX:

        from detectors.onnx_detector import ONNXDetector

        return ONNXDetector()

    if engine == config.OPENVINO:

        from detectors.openvino_detector import OpenVINODetector

        return OpenVINODetector()

    if engine == config.OPENCV:

        from detectors.opencv_detector import OpenCVDetector

        return OpenCVDetector()

    raise ValueError(
        f"Unsupported engine: {engine}"
    )


def create_frame_source():
    """
    Create image frame source for validation.
    """

    return FrameSource(
        input_mode=config.IMAGE,
        image_path=config.INPUT_IMAGE,
        webcam_index=config.WEBCAM_INDEX,
        rtsp_url=config.RTSP_URL,
    )


def validate_backend(engine):
    """
    Validate one backend using the shared
    raw-frame processing pipeline.
    """

    print("\n" + "=" * 60)
    print(f"Backend : {engine}")
    print("=" * 60)

    detector = create_detector(
        engine
    )

    processor = FrameProcessor(
        detector
    )

    with create_frame_source() as source:

        success, frame = source.read()

        if not success:

            raise RuntimeError(
                f"Failed to read frame for {engine}."
            )

        result = processor.process(
            frame
        )

    print(
        f"Frame ID    : "
        f"{result['frame_id']}"
    )

    print(
        f"Persons     : "
        f"{result['persons']}"
    )

    print(
        f"Frame Shape : "
        f"{result['annotated_frame'].shape}"
    )

    timings = result["timings"]

    print(
        f"Preprocess  : "
        f"{timings['preprocess_ms']:.3f} ms"
    )

    print(
        f"Inference   : "
        f"{timings['inference_ms']:.3f} ms"
    )

    print(
        f"Postprocess : "
        f"{timings['postprocess_ms']:.3f} ms"
    )

    print(
        f"Draw        : "
        f"{timings['draw_ms']:.3f} ms"
    )

    return result


def main():
    """
    Validate all detector backends.
    """

    expected_detections = {
        config.PYTORCH: 16,
        config.ONNX: 18,
        config.OPENVINO: 18,
        config.OPENCV: 18,
    }

    successful_backends = 0

    for engine in BACKENDS:

        result = validate_backend(
            engine
        )

        expected = expected_detections[
            engine
        ]

        actual = result["persons"]

        if actual != expected:

            raise RuntimeError(
                "Detection count mismatch: "
                f"{engine} | "
                f"Expected {expected} | "
                f"Actual {actual}"
            )

        timings = result["timings"]

        for timing_name in [
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "draw_ms",
        ]:

            if timings[timing_name] <= 0:

                raise RuntimeError(
                    "Invalid frame timing: "
                    f"{engine} | "
                    f"{timing_name} = "
                    f"{timings[timing_name]}"
                )

        successful_backends += 1

    print("\n" + "=" * 60)
    print("Shared Frame Processor Validation")
    print("=" * 60)

    print(
        f"Successful Backends : "
        f"{successful_backends}/"
        f"{len(BACKENDS)}"
    )

    print("Status              : PASS")

    print("=" * 60)


if __name__ == "__main__":
    main()