"""
Detection Post Processor

Project:
LowEndPC-PersonDetection

Author:
Deep Chakraborty
"""

import config
import numpy as np
import cv2


class PostProcessor:

    def __init__(self):

        self.confidence = config.CONFIDENCE
        self.iou = config.IOU
        self.person_class = config.PERSON_CLASS

    # --------------------------------------------------

    def process(self, prediction):
        """
        Complete postprocessing pipeline.

        Current Pipeline

        Decode
            ↓
        Filter
            ↓
        Convert Boxes
            ↓
        NMS
        """

        boxes, scores = self.decode(prediction)

        boxes, confidences, class_ids = self.filter(
            boxes,
            scores
        )

        boxes = self.convert_boxes(boxes)

        boxes, confidences, class_ids = self.nms(
            boxes,
            confidences,
            class_ids
        )

        return (
            boxes,
            confidences,
            class_ids
        )

    # --------------------------------------------------

    def decode(self, prediction):
        """
        Raw tensor -> boxes + scores
        """
        predictions = prediction.squeeze(0).T

        boxes = predictions[:, :4]

        scores = predictions[:, 4:]

        return boxes, scores

    # --------------------------------------------------

    def filter(
        self,
        boxes,
        scores
    ):
        """
        Person class + confidence threshold
        """
        class_ids = scores.argmax(axis=1)

        confidences = scores.max(axis=1)

        mask = (
            (class_ids == self.person_class)
            &
            (confidences >= self.confidence)
        )

        boxes = boxes[mask]

        confidences = confidences[mask]

        class_ids = class_ids[mask]

        return (
            boxes,
            confidences,
            class_ids
        )
    # --------------------------------------------------

    def convert_boxes(self, boxes):
        """
        Convert YOLO bounding boxes from
        (x_center, y_center, width, height)
        to
        (x1, y1, x2, y2)
        """

        boxes = boxes.copy()

        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

    # --------------------------------------------------

    def nms(
            self,
            boxes,
            scores,
            class_ids
        ):
            """
            Apply Non-Maximum Suppression.

            Parameters
            ----------
            boxes : numpy.ndarray
                Shape (N, 4) in xyxy format.

            scores : numpy.ndarray
                Confidence scores.

            class_ids : numpy.ndarray

            Returns
            -------
            tuple
                boxes,
                scores,
                class_ids
            """

            if len(boxes) == 0:

                return (
                    boxes,
                    scores,
                    class_ids
                )

            boxes_cv = []

            for box in boxes:

                x1, y1, x2, y2 = box

                boxes_cv.append([
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ])

            indices = cv2.dnn.NMSBoxes(
                boxes_cv,
                scores.tolist(),
                self.confidence,
                self.iou
            )

            if len(indices) == 0:

                return (
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int32)
                )

            indices = np.array(indices).flatten()

            return (
                boxes[indices],
                scores[indices],
                class_ids[indices]
            )

    # --------------------------------------------------

    def normalize(
        self,
        boxes,
        scores,
        class_ids
    ):
        """
        Return framework-standard detections.
        """
        raise NotImplementedError