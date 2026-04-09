import numpy as np
import cv2

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