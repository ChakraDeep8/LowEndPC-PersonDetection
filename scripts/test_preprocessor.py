import cv2

import config
from pipeline.preprocessor import Preprocessor

image = cv2.imread(str(config.INPUT_IMAGE))

processor = Preprocessor()
from detectors.onnx_detector import ONNXDetector
detector = ONNXDetector()

# from detectors.pytorch_detector import PyTorchDetector
# detector = PyTorchDetector()
# tensor = processor.process(image)

# print("Shape :", tensor.shape)
# print("Type  :", tensor.dtype)
# print("Min   :", tensor.min())
# print("Max   :", tensor.max())

# data = detector.preprocessor.process(image)

# print(data["tensor"].shape)
# print(data["original_shape"])
# print(data["resized_shape"])
# print(data["scale"])
# print(data["pad"])

tensor = detector.preprocess(image)

print(detector.preprocess_metadata)