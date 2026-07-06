import cv2

import config
from pipeline.preprocessor import Preprocessor

image = cv2.imread(str(config.INPUT_IMAGE))

processor = Preprocessor()

tensor = processor.process(image)

print("Shape :", tensor.shape)
print("Type  :", tensor.dtype)
print("Min   :", tensor.min())
print("Max   :", tensor.max())