"""
It will measure the performance of various computer vision operations:

CPU information
RAM usage
OpenCV version
AVX2 availability
Image read time
Image resize time
Color conversion time
Gaussian blur time
Canny edge detection time
FPS for a simple processing pipeline """

import cv2
import time
import psutil
import platform
import cpuinfo

IMAGE = "images/person.jpg"

print("=" * 50)
print("LowEndPC-PersonDetection Benchmark")
print("=" * 50)

print(f"CPU : {cpuinfo.get_cpu_info()["brand_raw"]}")
print(f"OpenCV : {cv2.__version__}")
print(f"Logical CPUs : {psutil.cpu_count()}")
print(f"RAM : {round(psutil.virtual_memory().total/1024**3,2)} GB")

img = cv2.imread(IMAGE)

if img is None:
    raise Exception("Image not found!")

start = time.perf_counter()

for _ in range(100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,100,200)
    resize = cv2.resize(edges,(640,640))

end = time.perf_counter()

elapsed = end-start

fps = 100/elapsed

print()
print(f"Elapsed : {elapsed:.3f} sec")
print(f"FPS : {fps:.2f}")

print()
print("CPU Usage:", psutil.cpu_percent(interval=1),"%")
print("RAM Usage:", psutil.virtual_memory().percent,"%")