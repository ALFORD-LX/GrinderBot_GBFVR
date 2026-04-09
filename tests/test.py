"""
detect_characters.py
--------------------
Usage:
    python detect_characters.py <game_screenshot> [sprite1 sprite2 ...]

Examples:
    # No sprites — runs background-subtraction / edge-blob heuristic only
    python detect_characters.py screenshot.png

    # With reference sprites
    python detect_characters.py screenshot.png idle.png crouch.png jump.png

Controls (OpenCV window):
    Q / ESC  — quit
    S        — save annotated image to  detected_output.png
    D        — toggle debug edge view
"""

import sys
import cv2
import numpy as np


# ─────────────────────────────────────────────
# Tunable parameters
# ─────────────────────────────────────────────
SCALES        = np.linspace(0.4, 1.6, 25)   # scale range for multiscale search
MATCH_THRESH  = 0.40                          # min confidence to draw a box
NMS_OVERLAP   = 0.3                           # non-max suppression IoU threshold
CANNY_SIGMA   = 0.33                          # auto-Canny sensitivity
DILATE_ITERS  = 1                             # edge dilation (pose forgiveness)
MIN_BLOB_AREA = 1200                          # min px² for heuristic blobs


# ─────────────────────────────────────────────
# Edge helpers
# ─────────────────────────────────────────────
def auto_canny(gray: np.ndarray, sigma: float = CANNY_SIGMA) -> np.ndarray:
    """Canny with thresholds derived from image median."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median  = np.median(blurred)
    low     = int(max(0,   (1.0 - sigma) * median))
    high    = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(blurred, low, high)


def to_edges(img: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=DILATE_ITERS)


# ─────────────────────────────────────────────
# Non-maximum suppression
# ─────────────────────────────────────────────
def nms(boxes: list, scores: list, iou_thresh: float = NMS_OVERLAP) -> list:
    """Remove overlapping boxes, keeping the highest-scoring one."""
    if not boxes:
        return []
    boxes  = np.array(boxes,  dtype=float)
    scores = np.array(scores, dtype=float)

    x1 = boxes[:, 0];  y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    kept  = []

    while order.size:
        i = order[0]
        kept.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]

    return [boxes[k].astype(int).tolist() for k in kept], [scores[k] for k in kept]


# ─────────────────────────────────────────────
# Multiscale edge template matching
# ─────────────────────────────────────────────
def multiscale_edge_match(frame_edges: np.ndarray,
                           tmpl_edges:  np.ndarray,
                           scales:      np.ndarray = SCALES):
    """Return (box, confidence, scale) for the best match."""
    fh, fw = frame_edges.shape
    th, tw = tmpl_edges.shape
    best_val, best_box, best_scale = -1, None, None

    for scale in scales:
        rw, rh = int(tw * scale), int(th * scale)
        if rw > fw or rh > fh or rw < 10 or rh < 10:
            continue

        resized = cv2.resize(tmpl_edges, (rw, rh))
        result  = cv2.matchTemplate(frame_edges, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val   = max_val
            best_box   = (*max_loc, rw, rh)
            best_scale = scale

    return best_box, best_val, best_scale


# ─────────────────────────────────────────────
# Heuristic: find character-like blobs from edges alone
# (used when no reference sprites are supplied)
# ─────────────────────────────────────────────
def heuristic_blobs(frame: np.ndarray, min_area: int = MIN_BLOB_AREA):
    """
    Segment moving / high-contrast regions that look like characters.
    Works purely on the screenshot — no reference sprite needed.
    Strategy:
      1. Get edges.
      2. Morphologically close gaps → filled silhouettes.
      3. Filter contours by area and aspect ratio typical of humanoid sprites.
    """
    edges   = to_edges(frame)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / (w + 1e-6)
        # Characters are generally taller than wide (0.8 – 4.0)
        if 0.6 <= aspect <= 4.5:
            score = min(area / 30000, 0.99)   # pseudo-confidence
            results.append(((x, y, w, h), score))

    if not results:
        return [], []

    boxes, scores = zip(*results)
    return nms(list(boxes), list(scores))


# ─────────────────────────────────────────────
# Main detection pipeline
# ─────────────────────────────────────────────
def detect(frame: np.ndarray, sprites: dict) -> tuple:
    """
    Returns:
        annotated  — BGR image with boxes drawn
        debug_img  — edge map for debug view
        detections — list of dicts with box / label / confidence
    """
    frame_edges = to_edges(frame)
    detections  = []

    if sprites:
        raw_boxes, raw_scores, raw_labels = [], [], []

        for label, tmpl in sprites.items():
            tmpl_edges = to_edges(tmpl)
            box, conf, scale = multiscale_edge_match(frame_edges, tmpl_edges)
            if box and conf >= MATCH_THRESH:
                raw_boxes.append(box)
                raw_scores.append(conf)
                raw_labels.append(f"{label} {conf:.2f}")

        if raw_boxes:
            kept_boxes, kept_scores = nms(raw_boxes, raw_scores)
            # Re-attach labels (closest score wins)
            for kb, ks in zip(kept_boxes, kept_scores):
                idx = int(np.argmin([abs(ks - s) for s in raw_scores]))
                detections.append({"box": kb, "label": raw_labels[idx],
                                   "confidence": ks, "source": "template"})

    else:
        # No sprites — fall back to heuristic blobs
        kept_boxes, kept_scores = heuristic_blobs(frame)
        for kb, ks in zip(kept_boxes, kept_scores):
            detections.append({"box": kb, "label": f"? {ks:.2f}",
                               "confidence": ks, "source": "heuristic"})

    # Draw results
    annotated = frame.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det["box"]
        color = (0, 255, 80) if det["source"] == "template" else (0, 180, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label_bg_y = max(y - 22, 0)
        (tw, th), _ = cv2.getTextSize(det["label"], cv2.FONT_HERSHEY_SIMPLEX,
                                       0.55, 1)
        cv2.rectangle(annotated, (x, label_bg_y), (x + tw + 4, label_bg_y + th + 4),
                      color, -1)
        cv2.putText(annotated, det["label"], (x + 2, label_bg_y + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay legend
    mode_txt = f"Sprites: {len(sprites)}" if sprites else "Mode: heuristic (no sprites)"
    cv2.putText(annotated, mode_txt, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Detections: {len(detections)}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "S=save  D=debug  Q=quit", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Debug: colour edge map
    debug_img = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2BGR)
    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 80), 2)

    return annotated, debug_img, detections


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    screenshot_path = sys.argv[1]
    sprite_paths    = sys.argv[2:]

    frame = cv2.imread(screenshot_path)
    if frame is None:
        print(f"[ERROR] Could not load image: {screenshot_path}")
        sys.exit(1)

    sprites = {}
    for path in sprite_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not load sprite: {path} — skipping")
            continue
        name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
        sprites[name] = img
        print(f"[INFO] Loaded sprite: {name}")

    print(f"[INFO] Processing: {screenshot_path}")
    annotated, debug_img, detections = detect(frame, sprites)

    for d in detections:
        x, y, w, h = d["box"]
        print(f"  [{d['source']}] {d['label']:30s}  box=({x},{y},{w},{h})")

    if not detections:
        print("  No characters detected — try lowering MATCH_THRESH or adding sprites.")

    # Display
    show_debug = False
    cv2.namedWindow("Character Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Character Detection", annotated)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):          # Q or ESC — quit
            break

        elif key == ord('s'):              # S — save
            out = "detected_output.png"
            cv2.imwrite(out, annotated)
            print(f"[INFO] Saved → {out}")

        elif key == ord('d'):              # D — toggle debug
            show_debug = not show_debug
            cv2.imshow("Character Detection",
                       debug_img if show_debug else annotated)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()