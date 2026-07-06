import cv2

import config
from detectors.onnx_detector import ONNXDetector

detector = ONNXDetector()

image = cv2.imread(str(config.INPUT_IMAGE))

tensor = detector.preprocess(image)

output = detector.inference(tensor)

print("\nOutput Shape :", output.shape)
print("Output Type  :", output.dtype)
print("Min Value    :", output.min())
print("Max Value    :", output.max())

# result = detector.postprocess(output)

# boxes, confidences, class_ids = result

# print(f"Persons: {len(boxes)}")
# print(f"Boxes: {boxes.shape}")
# print(f"Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")

boxes, confidences, class_ids = detector.postprocess(output)

print("\nAfter Postprocess")

print("Boxes Shape :", boxes.shape)
print("Confidences :", confidences)
print("Class IDs   :", class_ids)

print("Total Persons :", len(boxes))