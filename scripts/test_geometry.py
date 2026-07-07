from utils.geometry import calculate_iou

box1 = [100, 100, 200, 200]
box2 = [100, 100, 200, 200]

print("IoU (Identical):", calculate_iou(box1, box2))

box3 = [300, 300, 400, 400]

print("IoU (No Overlap):", calculate_iou(box1, box3))

box4 = [150, 150, 250, 250]

print("IoU (Partial):", calculate_iou(box1, box4))