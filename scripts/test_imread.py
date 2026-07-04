import time
import cv2

image_path = "images/person.jpg"

for i in range(10):
    start = time.perf_counter()

    image = cv2.imread(image_path)

    elapsed = (time.perf_counter() - start) * 1000

    print(f"Run {i+1:02d}: {elapsed:.3f} ms")