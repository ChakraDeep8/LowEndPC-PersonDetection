"""
LowEndPC-PersonDetection

Main Entry Point

main.py
        │
        ▼
Isolated Backend Runner
        │
        ├── PyTorch Process
        ├── ONNX Runtime Process
        ├── OpenVINO Process
        └── OpenCV DNN Process
                │
                ▼
        BenchmarkRunner
                │
                ├── Timer
                ├── Logger
                ├── Profiler
                ├── Statistics
                └── SystemInfo
"""

import argparse
import subprocess
import sys
import cv2
from benchmark.benchmark_runner import BenchmarkRunner
from pipeline.frame_source import FrameSource
from pipeline.frame_processor import FrameProcessor
import config


# --------------------------------------------------


def create_detector(engine):
    """
    Create detector for the selected backend.

    Local imports keep backend runtimes isolated
    inside their respective benchmark processes.
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


# --------------------------------------------------

def create_frame_source():
    """
    Create the configured frame source.
    """

    return FrameSource(
        input_mode=config.INPUT_MODE,
        image_path=config.INPUT_IMAGE,
        webcam_index=config.WEBCAM_INDEX,
        rtsp_url=config.RTSP_URL,
    )


# --------------------------------------------------

def run_frame_source():
    """
    Run the configured frame source.

    This execution path validates unified frame
    acquisition independently from benchmarking.
    """

    print("\n" + "=" * 60)
    print("Frame Source")
    print("=" * 60)

    print(
        f"\nInput Mode : {config.INPUT_MODE}"
    )

    frame_count = 0

    with create_frame_source() as source:

        while True:

            success, frame = source.read()

            if not success:
                break

            frame_count += 1

            height, width = frame.shape[:2]

            print(
                f"Frame {frame_count} : "
                f"{width}x{height}"
            )

    print(
        f"\nFrames Read : {frame_count}"
    )

    print("=" * 60)

def run_frame_source():
    """
    Run the configured shared frame pipeline.
    """

    print("\n" + "=" * 60)
    print("Shared Frame Processing Pipeline")
    print("=" * 60)

    print(
        f"\nInput Mode : {config.INPUT_MODE}"
    )

    print(
        f"Backend    : {config.FRAME_ENGINE}"
    )

    detector = create_detector(
        config.FRAME_ENGINE
    )

    processor = FrameProcessor(
        detector
    )

    frame_count = 0

    last_result = None

    with create_frame_source() as source:

        while True:

            success, frame = source.read()

            if not success:
                break

            result = processor.process(
                frame
            )
            last_result = result
            frame_count += 1

            annotated_frame = result[
                "annotated_frame"
            ]

            if config.INPUT_MODE == config.IMAGE:

                timings = result["timings"]

                print(
                    f"\nFrame       : "
                    f"{result['frame_id']}"
                )

                print(
                    f"Persons     : "
                    f"{result['persons']}"
                )

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

            if config.INPUT_MODE == config.WEBCAM:

                cv2.putText(
                    annotated_frame,
                    (
                        "Persons: "
                        f"{result['persons']}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    annotated_frame,
                    (
                        "FPS: "
                        f"{result['frame_fps']:.2f}"
                    ),
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(
                    "LowEndPC Person Detection",
                    annotated_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    print(
                        "\nWebcam termination "
                        "requested."
                    )

                    break

    if config.INPUT_MODE == config.WEBCAM:

        cv2.destroyAllWindows()

    print("\n" + "=" * 60)

    print(
        f"Frames Processed : {frame_count}"
    )

    if last_result is not None:

        print(
            f"Average FPS      : "
            f"{last_result['average_fps']:.2f}"
        )

    print(
        "Shared Frame Pipeline Completed"
    )

    print("=" * 60)

# --------------------------------------------------

def get_benchmark_notes(engine):
    """
    Return benchmark notes for the selected backend.
    """

    notes = {

        config.PYTORCH:
            "PyTorch FP32 Isolated Baseline",

        config.ONNX:
            "ONNX Runtime FP32 Isolated Baseline",

        config.OPENVINO:
            "OpenVINO FP32 Isolated Baseline",

        config.OPENCV:
            "OpenCV DNN FP32 Isolated Baseline"

    }

    return notes[engine]


# --------------------------------------------------


def run_single_benchmark(engine, trial=None):
    """
    Run one backend benchmark in the current process.
    """

    print("\n" + "=" * 60)

    print(
        f"Isolated Benchmark Backend : {engine}"
    )

    if trial is not None:

        print(
            f"Isolated Trial             : {trial}"
        )

    print("=" * 60 + "\n")

    detector = create_detector(
        engine
    )

    runner = BenchmarkRunner(
        detector
    )

    runner.warmup(
        config.INPUT_IMAGE
    )

    notes = get_benchmark_notes(
        engine
    )

    if trial is not None:

        notes = (
            f"{notes} | Trial {trial}"
        )

    report = runner.benchmark(
        config.INPUT_IMAGE,
        notes=notes
    )

    runner.print_report(
        report
    )


# --------------------------------------------------


def run_isolated_benchmarks():
    """
    Run repeated isolated benchmark trials.

    Every backend trial executes inside a fresh Python
    process to prevent runtime and process-memory
    contamination between benchmark experiments.
    """

    backends = [

        config.PYTORCH,
        config.ONNX,
        config.OPENVINO,
        config.OPENCV

    ]

    total_trials = (
        len(backends)
        * config.ISOLATED_TRIALS
    )

    completed_trials = 0

    print("\n" + "=" * 60)
    print("Repeated Isolated Backend Benchmark")
    print("=" * 60)

    print(
        f"\nBackends          : {len(backends)}"
    )

    print(
        f"Trials Per Backend: "
        f"{config.ISOLATED_TRIALS}"
    )

    print(
        f"Total Processes   : {total_trials}"
    )

    for engine in backends:

        print("\n" + "=" * 60)

        print(
            f"Backend : {engine}"
        )

        print("=" * 60)

        for trial in range(
            1,
            config.ISOLATED_TRIALS + 1
        ):

            print(
                f"\nStarting Trial "
                f"{trial}/"
                f"{config.ISOLATED_TRIALS}"
            )

            print("-" * 60)

            command = [

                sys.executable,

                str(
                    config.ROOT /
                    "main.py"
                ),

                "--engine",

                engine,

                "--trial",

                str(trial)

            ]

            result = subprocess.run(
                command,
                cwd=str(config.ROOT)
            )

            if result.returncode != 0:

                raise RuntimeError(
                    "Backend benchmark failed: "
                    f"{engine} | "
                    f"Trial {trial}"
                )

            completed_trials += 1

            print(
                f"\nCompleted Trial "
                f"{trial}/"
                f"{config.ISOLATED_TRIALS}"
            )

            print(
                f"Overall Progress : "
                f"{completed_trials}/"
                f"{total_trials}"
            )

        print(
            f"\nBackend completed: {engine}"
        )

    print("\n" + "=" * 60)

    print(
        "Repeated Isolated Benchmarks Completed"
    )

    print(
        f"Successful Processes : "
        f"{completed_trials}/"
        f"{total_trials}"
    )

    print("=" * 60)


# --------------------------------------------------


def run_validation():
    """
    Validate backend numerical equivalence.
    """

    from detectors.onnx_detector import ONNXDetector
    from detectors.opencv_detector import OpenCVDetector

    print(
        "\nRunning Backend Validation...\n"
    )

    reference_detector = ONNXDetector()

    target_detector = OpenCVDetector()

    runner = BenchmarkRunner(
        reference_detector
    )

    runner.validate(

        reference_detector=reference_detector,

        target_detector=target_detector,

        image_path=config.INPUT_IMAGE

    )


# --------------------------------------------------


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "LowEndPC Person Detection Benchmark"
        )
    )

    parser.add_argument(

        "--engine",

        choices=[

            config.PYTORCH,
            config.ONNX,
            config.OPENVINO,
            config.OPENCV

        ],

        default=None,

        help=(
            "Run one backend benchmark "
            "in the current process."
        )

    )

    parser.add_argument(

        "--trial",

        type=int,

        default=None,

        help=(
            "Isolated benchmark trial number."
        )

    )

    return parser.parse_args()


# --------------------------------------------------


def main():

    args = parse_arguments()

    # --------------------------------------------------
    # Isolated Child Process
    # --------------------------------------------------

    if args.engine is not None:

        run_single_benchmark(
            args.engine,
            args.trial
        )

        return

    # --------------------------------------------------
    # Main Run Mode
    # --------------------------------------------------

    if config.RUN_MODE == config.BENCHMARK:

        run_isolated_benchmarks()

    elif config.RUN_MODE == config.VALIDATION:

        run_validation()

    elif config.RUN_MODE == config.FRAME_SOURCE:

        run_frame_source()

    else:

        raise ValueError(
            f"Unsupported run mode: "
            f"{config.RUN_MODE}"
        )


# --------------------------------------------------


if __name__ == "__main__":

    main()