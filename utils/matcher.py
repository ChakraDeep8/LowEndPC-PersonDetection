"""
Detection Matcher

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

from utils.geometry import calculate_iou


def match_detections(
    reference_detections,
    target_detections,
    iou_threshold=0.5
):
    """
    Match detections using IoU.

    Parameters
    ----------
    reference_detections : list
        Usually PyTorch detections.

    target_detections : list
        Usually ONNX detections.

    iou_threshold : float

    Returns
    -------
    list
        Matched detections.
    """

    matches = []

    used_target = set()

    for reference in reference_detections:

        best_iou = 0.0
        best_index = None

        for index, target in enumerate(target_detections):

            if index in used_target:
                continue

            iou = calculate_iou(
                reference["bbox"],
                target["bbox"]
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if (
            best_index is not None
            and
            best_iou >= iou_threshold
        ):

            matches.append({

                "reference": reference,

                "target": target_detections[best_index],

                "iou": best_iou

            })

            used_target.add(best_index)

    return matches