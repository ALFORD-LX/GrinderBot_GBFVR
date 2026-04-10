from GrinderBot import CharacterFinder
import cv2
from time import perf_counter

finder = CharacterFinder()
frame = cv2.imread("tests/test.png")

startT = perf_counter()
matches = finder.find(frame)
endT = perf_counter()

# Draw results on frame
annotated = frame.copy()
colors = [(0, 255, 80), (0, 180, 255), (255, 100, 0), (180, 0, 255)]  # one per match

for i, m in enumerate(matches):
    print(f"{m['character']} at {m['position']} ({m['confidence']:.2f})")
    color = colors[i % len(colors)]
    x, y, w, h = m["box"]

    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    label = f"{m['character']} {m['confidence']:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_y = max(y - 22, 0)
    cv2.rectangle(annotated, (x, label_y), (x + tw + 4, label_y + th + 4), color, -1)
    cv2.putText(annotated, label, (x + 2, label_y + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

print(f"Time to detect 1 frame: {endT - startT:.4f}s")

cv2.namedWindow("Character Finder", cv2.WINDOW_NORMAL)
cv2.imshow("Character Finder", annotated)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('s'):
        cv2.imwrite("finder_output.png", annotated)
        print("[INFO] Saved → finder_output.png")

cv2.destroyAllWindows()