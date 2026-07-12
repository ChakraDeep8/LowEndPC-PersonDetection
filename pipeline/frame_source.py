from pathlib import Path

import cv2


class FrameSource:
    IMAGE = "image"
    WEBCAM = "webcam"
    RTSP = "rtsp"

    SUPPORTED_MODES = {
        IMAGE,
        WEBCAM,
        RTSP,
    }

    def __init__(
        self,
        input_mode,
        image_path=None,
        webcam_index=0,
        rtsp_url=None,
    ):
        self.input_mode = input_mode
        self.image_path = (
            Path(image_path)
            if image_path is not None
            else None
        )
        self.webcam_index = webcam_index
        self.rtsp_url = rtsp_url

        self.capture = None
        self.image = None
        self.image_consumed = False
        self.is_open = False

    def open(self):
        """
        Open the configured frame source.
        """

        if self.input_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported input mode: {self.input_mode}"
            )

        if self.is_open:
            return

        if self.input_mode == self.IMAGE:
            self._open_image()

        elif self.input_mode == self.WEBCAM:
            self._open_webcam()

        elif self.input_mode == self.RTSP:
            self._open_rtsp()

        self.is_open = True

    def _open_image(self):
        """
        Load a single image frame.
        """

        if self.image_path is None:
            raise ValueError(
                "Image path is required for image mode."
            )

        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {self.image_path}"
            )

        self.image = cv2.imread(
            str(self.image_path)
        )

        if self.image is None:
            raise RuntimeError(
                f"Failed to read image: {self.image_path}"
            )

        self.image_consumed = False

    def _open_webcam(self):
        """
        Open a webcam capture source.
        """

        self.capture = cv2.VideoCapture(
            self.webcam_index
        )

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None

            raise RuntimeError(
                "Failed to open webcam "
                f"at index {self.webcam_index}."
            )

    def _open_rtsp(self):
        """
        Open an RTSP capture source.
        """

        if not self.rtsp_url:
            raise ValueError(
                "RTSP URL is required for RTSP mode."
            )

        self.capture = cv2.VideoCapture(
            self.rtsp_url
        )

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None

            raise RuntimeError(
                "Failed to open RTSP stream."
            )

    def read(self):
        """
        Read the next frame.

        Returns:
            tuple:
                success (bool)
                frame (numpy.ndarray | None)
        """

        if not self.is_open:
            raise RuntimeError(
                "Frame source is not open."
            )

        if self.input_mode == self.IMAGE:

            if self.image_consumed:
                return False, None

            self.image_consumed = True

            return True, self.image

        success, frame = self.capture.read()

        if not success or frame is None:
            return False, None

        return True, frame

    def release(self):
        """
        Release the frame source.
        """

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.image = None
        self.image_consumed = False
        self.is_open = False

    def __enter__(self):
        self.open()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.release()