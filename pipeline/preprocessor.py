"""
Image Preprocessor

Project:
LowEndPC-PersonDetection

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

    def process(self, image):

        """
        Convert OpenCV image into
        backend-ready tensor.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Shape:
            (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        """

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

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

        image = np.expand_dims(
            image,
            axis=0
        )

        return image