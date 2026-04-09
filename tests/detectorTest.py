from GrinderBot import SpriteDetector
import cv2
from time import perf_counter
import cProfile
import pstats

startT = perf_counter()
detector = SpriteDetector("char1")
endT = perf_counter()
print(f"Time to load: {endT - startT:.4f}s")

frame = cv2.imread("tests/test.png")

# Warmup call so any lazy init doesn't skew results
detector.detect(frame)

# Profile over 10 runs
with cProfile.Profile() as pr:
    for _ in range(10):
        detector.detect(frame)

stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
stats.print_stats(20)

# Still show a single-run time
startT = perf_counter()
result = detector.detect(frame)
endT = perf_counter()

if result["found"]:
    print(f"Found at {result['position']} with confidence {result['confidence']:.2f}")
else:
    print("Sprite not found")
print(f"Time to detect 1 frame: {endT - startT:.4f}s")