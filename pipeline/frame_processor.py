"""
Shared Frame Processor

Project:
LowEndPC-PersonDetection

Frame
    │
    ▼
Preprocess
    │
    ▼
Inference
    │
    ▼
Postprocess
    │
    ▼
Draw
    │
    ▼
Frame Result

Author:
Deep Chakraborty
"""

import time
from benchmark.timer import Timer
from detectors.base_detector import BaseDetector


class FrameProcessor:

    def __init__(
        self,
        detector
    ):
        """
        Initialize the shared frame processor.
        """

        if not isinstance(
            detector,
            BaseDetector,
        ):
            raise TypeError(
                "detector must inherit BaseDetector."
            )
        self.detector = detector

        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.last_frame_time = None

    # --------------------------------------------------

    def process(
        self,
        frame,
    ):
        """
        Process one in-memory frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR input frame.

        Returns
        -------
        dict
            Frame processing result.
        """

        if frame is None:
            raise ValueError(
                "Frame cannot be None."
            )

        if not hasattr(frame, "shape"):
            raise TypeError(
                "Frame must be a NumPy-compatible image."
            )

        if frame.size == 0:
            raise ValueError(
                "Frame cannot be empty."
            )

        self.frame_count += 1
        current_time = time.perf_counter()

        if self.last_frame_time is None:
            frame_fps = 0.0

        else:
            frame_interval = (
                current_time
                - self.last_frame_time
            )

            frame_fps = (
                1.0 / frame_interval
                if frame_interval > 0
                else 0.0
            )

        self.last_frame_time = current_time

        # --------------------------------------------------
        # Preprocess
        # --------------------------------------------------

        with Timer("Preprocess") as timer:

            processed = self.detector.preprocess(
                frame
            )

        self.detector.timings[
            "preprocess_ms"
        ] = timer.elapsed_ms

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        with Timer("Inference") as timer:

            predictions = self.detector.inference(
                processed
            )

        self.detector.timings[
            "inference_ms"
        ] = timer.elapsed_ms

        # --------------------------------------------------
        # Postprocess
        # --------------------------------------------------

        with Timer("Postprocess") as timer:

            detections = self.detector.postprocess(
                predictions
            )

        self.detector.timings[
            "postprocess_ms"
        ] = timer.elapsed_ms

        # --------------------------------------------------
        # Draw
        # --------------------------------------------------

        with Timer("Draw") as timer:

            annotated_frame = self.detector.draw(
                frame.copy(),
                detections,
            )

        self.detector.timings[
            "draw_ms"
        ] = timer.elapsed_ms

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        elapsed_time = (
            time.perf_counter()
            - self.start_time
        )

        average_fps = (
            self.frame_count / elapsed_time
            if elapsed_time > 0
            else 0.0
        )

        return {
            "frame_id": self.frame_count,
            "detections": detections,
            "persons": len(detections),
            "annotated_frame": annotated_frame,
            "timings": self.detector.timings.copy(),
            "frame_fps": frame_fps,
            "average_fps": average_fps,
        }