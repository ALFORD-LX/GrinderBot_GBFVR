from GrinderBot.sprite_detection import SpriteDetector
import cv2
from time import perf_counter

startT = perf_counter()
detector = SpriteDetector("char1")
endT = perf_counter()

print(f"Time to load: {endT - startT}")

startT = perf_counter()

frame = cv2.imread("test.png")

result = detector.detect(frame)

if result["found"]:
    print(f"Found at {result['position']} with confidence {result['confidence']:.2f}")
else:
    print("Sprite not found")
endT = perf_counter()


print(f"Time to detect 1 frame: {endT - startT}")