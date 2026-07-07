from utils.matcher import match_detections

pytorch = [

    {
        "bbox": [100, 100, 200, 200],
        "confidence": 0.90,
        "class_id": 0,
        "label": "person"
    },

    {
        "bbox": [300, 300, 400, 400],
        "confidence": 0.88,
        "class_id": 0,
        "label": "person"
    }

]

onnx = [

    {
        "bbox": [102, 101, 199, 201],
        "confidence": 0.89,
        "class_id": 0,
        "label": "person"
    },

    {
        "bbox": [301, 302, 399, 401],
        "confidence": 0.87,
        "class_id": 0,
        "label": "person"
    }

]

matches = match_detections(
    pytorch,
    onnx
)

print("Matches :", len(matches))

for index, match in enumerate(matches, start=1):

    print(f"\nMatch {index}")

    print("IoU :", round(match["iou"], 4))

    print("PyTorch :", match["reference"]["bbox"])

    print("ONNX    :", match["target"]["bbox"])