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

# # Profile over 10 runs
# with cProfile.Profile() as pr:
#     for _ in range(10):
#         detector.detect(frame)

# stats = pstats.Stats(pr)
# stats.sort_stats("cumulative")
# stats.print_stats(20)

# Still show a single-run time
startT = perf_counter()
result = detector.detect(frame)
endT = perf_counter()

if result["found"]:
    print(f"Found at {result['position']} with confidence {result['confidence']:.2f}")
else:
    print("Sprite not found")
print(f"Time to detect 1 frame: {endT - startT:.4f}s")

if result["found"]:
    x, y, w, h = result["box"]
    xp, yp = result["position"]
    confidence = result["confidence"]
    label = f"{detector.sprite_name}: {confidence:.2f}"

    # Draw rectangle (green)
    cv2.rectangle(frame, (xp, yp), (x + w, y + h), (0, 255, 0), 2)

    # Draw label background (optional)
    cv2.rectangle(frame, (x, y - 20), (x + len(label) * 10, y), (0, 255, 0), -1)

    # Draw text (white)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"Found at ({x}, {y}) with confidence {confidence:.2f}")
else:
    print("Sprite not found")

# Display the image
cv2.imshow("Detection Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
