"""
YOLO Output Decoder

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import numpy as np

import config


class YOLODecoder:

    def __init__(self):

        self.confidence = config.CONFIDENCE
        self.person_class = config.PERSON_CLASS

    # --------------------------------------------------

    def decode(self, output):

        """
        Decode raw YOLO output.

        Parameters
        ----------
        output : numpy.ndarray
            Shape: (1, 84, 8400)

        Returns
        -------
        list
            Normalized detections.
        """

        predictions = output.squeeze(0).T

        print(f"Decoded Shape : {predictions.shape}")

        boxes = predictions[:, :4]

        scores = predictions[:, 4:]

        print(f"Boxes Shape  : {boxes.shape}")
        print(f"Scores Shape : {scores.shape}")

        class_ids = scores.argmax(axis=1)

        confidences = scores.max(axis=1)

        mask = (
            (class_ids == self.person_class)
            &
            (confidences >= self.confidence)
        )

        boxes = boxes[mask]

        boxes_xyxy = boxes.copy()

        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        confidences = confidences[mask]

        class_ids = class_ids[mask]

        print(f"Person Detections : {len(boxes)}")

        return boxes_xyxy, confidences, class_ids