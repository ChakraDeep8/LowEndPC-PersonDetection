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

    def process(self, prediction, metadata):
        """
        Complete postprocessing pipeline.

        Decode
            ↓
        Filter
            ↓
        Convert Boxes
            ↓
        NMS
            ↓
        Normalize
        """

        boxes, scores = self.decode(prediction)

        boxes, confidences, class_ids = self.filter(
            boxes,
            scores
        )

        boxes = self.convert_boxes(boxes)

        boxes = self.restore_boxes(
            boxes,
            metadata
        )

        boxes, confidences, class_ids = self.nms(
            boxes,
            confidences,
            class_ids
        )

        detections = self.normalize(
            boxes,
            confidences,
            class_ids
        )

        return detections

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
                Bounding boxes in xyxy format.

            scores : numpy.ndarray
                Detection confidences.

            class_ids : numpy.ndarray

            Returns
            -------
            tuple
                boxes,
                scores,
                class_ids
            """

            import cv2
            import numpy as np

            if len(boxes) == 0:

                return (
                    boxes,
                    scores,
                    class_ids
                )

            nms_boxes = []

            for box in boxes:

                x1, y1, x2, y2 = box

                nms_boxes.append([
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ])

            indices = cv2.dnn.NMSBoxes(
                bboxes=nms_boxes,
                scores=scores.tolist(),
                score_threshold=self.confidence,
                nms_threshold=self.iou
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
            Convert detections into the framework-standard format.

            Returns
            -------
            list[dict]
            """

            detections = []

            for box, score, class_id in zip(
                boxes,
                scores,
                class_ids
            ):

                detections.append({

                    "bbox": box.tolist(),

                    "confidence": float(score),

                    "class_id": int(class_id),

                    "label": "person"

                })

            return detections
    

    def restore_boxes(
    self,
    boxes,
    metadata
     ):
        """
        Convert boxes from letterboxed image
        back to original image coordinates.
        """

        scale = metadata["scale"]

        pad_x, pad_y = metadata["pad"]

        original_height, original_width = metadata["original_shape"]

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y

        boxes /= scale
        
        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0,
            original_width
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0,
            original_height
        )

        return boxes
    