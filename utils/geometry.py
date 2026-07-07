"""
Geometry Utilities

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""


def calculate_iou(
    box1,
    box2
):
    """
    Calculate Intersection over Union (IoU).

    Parameters
    ----------
    box1 : list | tuple
        [x1, y1, x2, y2]

    box2 : list | tuple
        [x1, y1, x2, y2]

    Returns
    -------
    float
        IoU value between 0.0 and 1.0
    """

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])

    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (
        (x_right - x_left)
        * (y_bottom - y_top)
    )

    area1 = (
        (box1[2] - box1[0])
        * (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0])
        * (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union