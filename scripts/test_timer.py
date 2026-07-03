from benchmark.timer import Timer
import time

print("=" * 50)
print("Timer Test")
print("=" * 50)

with Timer("Sleep Benchmark") as timer:
    time.sleep(1.5)

print(timer)

print("\nDictionary Output")
print(timer.summary())