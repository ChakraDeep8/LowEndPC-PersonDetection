"""
Image Preprocessor

Project:
LowEndPC-PersonDetection

Algorithm:

Original Image
        │
        ▼
Get Original Size
        │
        ▼
Compute Scale
        │
        ▼
Resize (Keep Aspect Ratio)
        │
        ▼
Compute Padding
        │
        ▼
Pad to 640×640
        │
        ▼
BGR → RGB
        │
        ▼
float32 /255
        │
        ▼
HWC → CHW
        │
        ▼
Batch Dimension
        │
        ▼
Return Tensor + Metadata

Author:
Deep Chakraborty
"""

import cv2
import numpy as np

import config


class Preprocessor:

    def __init__(self):

        self.image_size = config.IMAGE_SIZE

    # --------------------------------------------------

    def letterbox(self, image):
        """
        Resize image while preserving aspect ratio.

        Returns
        -------
        resized_image
        scale
        (pad_x, pad_y)
        (new_width, new_height)
        """

        original_height, original_width = image.shape[:2]

        scale = min(
            self.image_size / original_width,
            self.image_size / original_height
        )

        new_width = int(round(original_width * scale))
        new_height = int(round(original_height * scale))

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        pad_width = self.image_size - new_width
        pad_height = self.image_size - new_height

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        return (
            padded,
            scale,
            (pad_left, pad_top),
            (new_width, new_height)
        )

    def process(self, image):
        """
        Convert image into backend-ready tensor.

        Returns
        -------
        dict
        """

        original_shape = image.shape[:2]

        image, scale, pad, resized_shape = self.letterbox(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(np.float32)

        image /= 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        tensor = np.expand_dims(
            image,
            axis=0
        )

        return {

            "tensor": tensor,

            "original_shape": original_shape,

            "resized_shape": resized_shape,

            "scale": scale,

            "pad": pad

        }