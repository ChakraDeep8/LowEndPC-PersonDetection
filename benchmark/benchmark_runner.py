"""
Instead of every detector implementing benchmarking:

PyTorch             ONNX            OpenVINO
    ↓                 ↓                 ↓
Benchmark         Benchmark         Benchmark


I separate them:

                 BenchmarkRunner
                       │
      ┌────────────────┼────────────────┐
      │                │                │
 PyTorchDetector  ONNXDetector  OpenVINODetector

This follows the Strategy Pattern. 
It's easy to plug different detectors into the same benchmarking engine.

Notice that BenchmarkRunner never imports Ultralytics.

It only understands normalized detections.

This separation makes the framework extensible.
"""

"""
Benchmark Runner

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""


from statistics import mean, median, stdev

from benchmark.timer import Timer
from benchmark.profiler import Profiler
from benchmark.logger import BenchmarkLogger
from benchmark.system_info import SystemInfo
import cv2
from pathlib import Path
import config
import gc

class BenchmarkRunner:

    def __init__(self, detector):

        self.detector = detector

        self.logger = BenchmarkLogger()

        self.profiler = Profiler()

        self.system = SystemInfo.collect()

    def warmup(self, image_path):

        print(f"\nWarmup ({config.WARMUP_RUNS} runs)")

        for i in range(config.WARMUP_RUNS):

            self.detector.detect(image_path)

            print(f"Warmup {i + 1}/{config.WARMUP_RUNS}")

    def benchmark(self, image_path, notes="BaseLine"):

        timings = []

        stage_timings = {

            "image_read_ms": [],

            "preprocess_ms": [],

            "inference_ms": [],

            "postprocess_ms": [],

            "draw_ms": [],

            "save_ms": []

        }

        cpu_usage = []
        ram_usage = []
        process_memory = []


        final_result = None

        print(f"\nBenchmark ({config.BENCHMARK_RUNS} runs)\n")


        for i in range(config.BENCHMARK_RUNS):

            gc.collect()

            with Timer(f"Run {i+1}") as timer:

                final_result = self.detector.detect(image_path)

            for key in stage_timings:

                stage_timings[key].append(
                    final_result["timings"][key]
                )            

            profile = self.profiler.snapshot()

            timings.append(timer.elapsed_ms)

            cpu_usage.append(profile["cpu_percent"])

            ram_usage.append(profile["memory_percent"])

            process_memory.append(profile["memory_used_mb"])

            print(
                f"Run {i+1:02d}: "
                f"{timer.elapsed_ms:7.2f} ms | "
                f"CPU {profile['cpu_percent']:5.1f}% | "
                f"RAM {profile['memory_percent']:5.1f}%"
            )
        
        image = cv2.imread(str(image_path))

        height, width = image.shape[:2]

        report = {

            # --------------------------------------------------
            # Backend Information
            # --------------------------------------------------

            "engine": self.detector.engine,
            "backend_version": self.detector.backend_version,
            "device": self.detector.device,

            # --------------------------------------------------
            # Model Information
            # --------------------------------------------------

            "model": self.detector.model_name,
            "model_format": self.detector.model_format,

            # --------------------------------------------------
            # Image Information
            # --------------------------------------------------

            "image": str(
                Path(image_path).relative_to(config.ROOT)
            ),

            "image_width": width,
            "image_height": height,
            "input_size": config.IMAGE_SIZE,

            # --------------------------------------------------
            # System Information
            # --------------------------------------------------

            "cpu_name": self.system["cpu"],
            "ram_gb": self.system["ram_gb"],

            # --------------------------------------------------
            # Timings
            # --------------------------------------------------

            "model_load_ms": self.detector.timings["model_load_ms"],

            "image_read_ms": mean(
                stage_timings["image_read_ms"]
            ),

            "preprocess_ms": mean(
                stage_timings["preprocess_ms"]
            ),

            "inference_ms": mean(
                stage_timings["inference_ms"]
            ),

            "postprocess_ms": mean(
                stage_timings["postprocess_ms"]
            ),

            "draw_ms": mean(
                stage_timings["draw_ms"]
            ),

            "save_ms": mean(
                stage_timings["save_ms"]
            ),

            # --------------------------------------------------
            # Benchmark Statistics
            # --------------------------------------------------

            "average_ms": mean(timings),

            "minimum_ms": min(timings),

            "maximum_ms": max(timings),

            "median_ms": median(timings),

            "std_ms": (
                stdev(timings)
                if len(timings) > 1
                else 0.0
            ),

            "fps": 1000 / mean(timings),

            # --------------------------------------------------
            # Resource Usage
            # --------------------------------------------------

            "cpu_percent": mean(cpu_usage),

            "ram_percent": mean(ram_usage),

            "process_memory": mean(process_memory),

            "threads": self.profiler.process.num_threads(),

            # --------------------------------------------------
            # Detection
            # --------------------------------------------------

            "detections": final_result["persons"],
            "success": True,

            # --------------------------------------------------
            # Miscellaneous
            # --------------------------------------------------

            "notes": notes,

            "output": final_result["output"]

        }

        self.save(report)

        return report
    
    def validate(
        self,
        reference_detector,
        target_detector,
        image_path
    ):
        """
        Validate two detector backends by comparing
        their final detections.
        """

        from utils.matcher import match_detections

        reference = reference_detector.detect(
            image_path
        )

        image = cv2.imread(str(image_path))

        height, width = image.shape[:2]

        print()

        print(f"Image Size : {width} x {height}")

        target = target_detector.detect(
            image_path
        )

        reference_detections = reference["detections"]
        target_detections = target["detections"]

        matches = match_detections(
            reference_detections,
            target_detections
        )

        matched_reference = {
            id(match["reference"])
            for match in matches
        }

        matched_target = {
            id(match["target"])
            for match in matches
        }

        unmatched_reference = [

            detection

            for detection in reference_detections

            if id(detection) not in matched_reference

        ]

        unmatched_target = [

            detection

            for detection in target_detections

            if id(detection) not in matched_target

        ]

        matched = len(matches)

        missing = (
            len(reference_detections)
            - matched
        )

        extra = (
            len(target_detections)
            - matched
        )

        if matched:

            average_iou = sum(
                match["iou"]
                for match in matches
            ) / matched


        else:

            average_iou = 0.0

        minimum_iou = min(
            (match["iou"] for match in matches),
            default=0.0
        )

        maximum_iou = max(
            (match["iou"] for match in matches),
            default=0.0
        )

        confidence_differences = [

            abs(
                match["reference"]["confidence"]
                -
                match["target"]["confidence"]
            )

            for match in matches

        ]

        average_confidence_difference = (

            sum(confidence_differences)
            / len(confidence_differences)

            if confidence_differences

            else 0.0

        )

        maximum_confidence_difference = (

            max(confidence_differences)

            if confidence_differences

            else 0.0

        )

        minimum_iou = min(
            (match["iou"] for match in matches),
            default=0.0
        )

        maximum_iou = max(
            (match["iou"] for match in matches),
            default=0.0
        )

        confidence_differences = [

            abs(
                match["reference"]["confidence"]
                -
                match["target"]["confidence"]
            )

            for match in matches

        ]

        average_confidence_difference = (

            sum(confidence_differences)
            / len(confidence_differences)

            if confidence_differences

            else 0.0

        )

        maximum_confidence_difference = (

            max(confidence_differences)

            if confidence_differences

            else 0.0

        )
        print()
        
        print("=" * 60)
        print("Backend Validation")
        print("=" * 60)

        print(
            f"Reference : {reference_detector.engine}"
        )

        print(
            f"Target    : {target_detector.engine}"
        )

        print()

        print(
            f"Reference Detections : {len(reference_detections)}"
        )

        print(
            f"Target Detections    : {len(target_detections)}"
        )

        print(
            f"Matched              : {matched}"
        )

        print(
            f"Missing              : {missing}"
        )

        print(
            f"Extra                : {extra}"
        )

        print()

        print(
            f"Average IoU          : {average_iou:.4f}"
        )

        print(
            f"Minimum IoU          : {minimum_iou:.4f}"
        )

        print(
            f"Maximum IoU          : {maximum_iou:.4f}"
        )

        print()

        print(
            f"Average Conf Diff    : "
            f"{average_confidence_difference:.4f}"
        )

        print(
            f"Maximum Conf Diff    : "
            f"{maximum_confidence_difference:.4f}"
        )

        print()

        if (
            matched == len(reference_detections)
            and
            average_iou >= 0.95
        ):

            print("Status : PASS")

        else:

            print("Status : REVIEW")

        print("=" * 60)

        if unmatched_reference:

            print()
            print("Unmatched Reference Detections")
            print("-" * 60)

            for index, detection in enumerate(
                unmatched_reference,
                start=1
            ):

                print(f"\n#{index}")

                print(
                    f"BBox       : {detection['bbox']}"
                )

                print(
                    f"Confidence : "
                    f"{detection['confidence']:.4f}"
                )

        if unmatched_target:

            print()
            print("Unmatched Target Detections")
            print("-" * 60)

            for index, detection in enumerate(
                unmatched_target,
                start=1
            ):

                print(f"\n#{index}")

                print(
                    f"BBox       : {detection['bbox']}"
                )

                print(
                    f"Confidence : "
                    f"{detection['confidence']:.4f}"
                )

                nearest, nearest_iou = self.nearest_detection(
                    detection,
                    reference_detections
                )

                print()

                if nearest is not None:

                    print(
                        f"Nearest IoU : {nearest_iou:.4f}"
                    )

                    print(
                        f"Nearest Confidence : "
                        f"{nearest['confidence']:.4f}"
                    )

                    if nearest_iou >= 0.5:

                        if nearest["confidence"] < detection["confidence"]:

                            reason = (
                                "Likely rejected by confidence "
                                "threshold."
                            )

                        else:

                            reason = (
                                "Likely NMS / numerical difference."
                            )

                    elif nearest_iou >= 0.3:

                        reason = (
                            "Borderline detection."
                        )

                    else:

                        reason = (
                            "Unique ONNX detection."
                        )

                    print(f"Reason : {reason}")

                else:

                    print("Reason : No nearby PyTorch detection.")


        return {

            "matched": matched,

            "missing": missing,

            "extra": extra,

            "average_iou": average_iou

        }
    
    def nearest_detection(
            self,
        detection,
        detections
    ):
        """
        Find the nearest detection using IoU.
        """

        from utils.geometry import calculate_iou

        best_detection = None
        best_iou = 0.0

        for candidate in detections:

            iou = calculate_iou(
                detection["bbox"],
                candidate["bbox"]
            )

            if iou > best_iou:

                best_iou = iou
                best_detection = candidate

        return best_detection, best_iou

    def save(self, report):

        self.logger.log(

        report["notes"],

        [

            # --------------------------------------------------
            # Backend Information
            # --------------------------------------------------

            report["engine"],
            report["backend_version"],
            report["device"],

            # --------------------------------------------------
            # Model Information
            # --------------------------------------------------

            report["model"],
            report["model_format"],

            # --------------------------------------------------
            # Image Information
            # --------------------------------------------------

            report["image"],
            report["image_width"],
            report["image_height"],
            report["input_size"],

            # --------------------------------------------------
            # System Information
            # --------------------------------------------------

            report["cpu_name"],
            report["ram_gb"],

            # --------------------------------------------------
            # Timings
            # --------------------------------------------------

            round(report["model_load_ms"], 3),
            round(report["image_read_ms"], 3),
            round(report["preprocess_ms"], 3),
            round(report["inference_ms"], 3),
            round(report["postprocess_ms"], 3),
            round(report["draw_ms"], 3),
            round(report["save_ms"], 3),
            round(report["average_ms"], 3),

            # --------------------------------------------------
            # Performance
            # --------------------------------------------------

            round(report["fps"], 2),

            round(report["cpu_percent"], 2),
            round(report["ram_percent"], 2),
            round(report["process_memory"], 2),

            self.profiler.process.num_threads(),

            # --------------------------------------------------
            # Detection
            # --------------------------------------------------

            report["detections"],
            report["success"]

        ]

    )
    def print_report(self, report):

        print()

        print("=" * 60)

        print("Benchmark Report")

        print("=" * 60)

        print(f"Engine            : {report['engine']}")

        print(f"Model             : {report['model']}")

        print(f"Model Load Time   : {report['model_load_ms']:.2f} ms")

        print(f"Image Read Time   : {report['image_read_ms']:.2f} ms")

        print(f"Preprocess Time   : {report['preprocess_ms']:.2f} ms")

        print(f"Inference Time    : {report['inference_ms']:.2f} ms")

        print(f"Postprocess Time  : {report['postprocess_ms']:.2f} ms")

        print(f"Draw Time         : {report['draw_ms']:.2f} ms")
        
        print(f"Save Time         : {report['save_ms']:.2f} ms")

        print(f"Average Latency   : {report['average_ms']:.2f} ms")

        print(f"Minimum Latency   : {report['minimum_ms']:.2f} ms")

        print(f"Maximum Latency   : {report['maximum_ms']:.2f} ms")

        print(f"Median Latency    : {report['median_ms']:.2f} ms")

        print(f"Std Deviation     : {report['std_ms']:.2f} ms")

        print(f"Average FPS       : {report['fps']:.2f}")

        print(f"CPU Usage         : {report['cpu_percent']:.2f} %")

        print(f"RAM Usage         : {report['ram_percent']:.2f} %")

        print(f"Process Memory    : {report['process_memory']:.2f} MB")

        print(f"Detections        : {report['detections']}")

        print(f"Output Image      : {report['output']}")

        print("=" * 60)