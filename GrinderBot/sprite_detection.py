import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
from GrinderBot.sprite_detection_util import *
from GrinderBot.constants import *

class SpriteDetector:
    def __init__(self, sprite_name: str) -> None:
        self.sprite_name = sprite_name
        self.sprites = self._load_sprites()

    def _load_sprites(self) -> dict:
        """Load all reference images from sprite_references/<sprite_name>/"""

        sprite_dir = Path(spriteRefrenceDirPath) / self.sprite_name
        if not sprite_dir.exists():
            raise FileNotFoundError(f"Sprite folder not found: {sprite_dir}")

        sprites = {}
        for path in sprite_dir.iterdir():
            if path.suffix.lower() in allowedImageTypes:
                img = cv2.imread(str(path))
                if img is not None:
                    sprites[path.stem] = to_edges(img)
                else:
                    print(f"[WARN] Could not load sprite: {path} — skipping")

        if not sprites:
            raise ValueError(f"No valid images found in: {sprite_dir}")

        print(f"[INFO] Loaded {len(sprites)} sprite(s) for '{self.sprite_name}'")
        return sprites

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a frame.

        Returns:
            {
                KEY_found:      bool,
                KEY_position:   (x, y) top-left corner or None,
                KEY_box:        (x, y, w, h) or None,
                KEY_confidence: float or None,
                KEY_label:      str or None,
            }
        """
        small = cv2.resize(frame, (0,0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        frame_edges = to_edges(small)
        raw_boxes, raw_scores, raw_labels = [], [], []

        for label, tmpl in self.sprites.items():
            box, conf, _ = multiscale_edge_match(frame_edges, tmpl)
            if box and conf >= MATCH_THRESH:
                raw_boxes.append(box)
                raw_scores.append(conf)
                raw_labels.append(f"{label} {conf:.2f}")

        if not raw_boxes:
            return {KEY_found: False, KEY_position: None, KEY_box: None,
                    KEY_confidence: None, KEY_label: None}

        kept_boxes, kept_scores = nms(raw_boxes, raw_scores)

        # Pick the highest-confidence kept detection
        best_idx = int(np.argmax(kept_scores))
        best_box = kept_boxes[best_idx]
        best_score = kept_scores[best_idx]

        # Recover the label whose raw score is closest to the kept score
        label_idx = int(np.argmin([abs(best_score - s) for s in raw_scores]))
        best_label = raw_labels[label_idx]
        
        x, y, w, h = best_box
        box = (int(x / DETECTION_SCALE), int(y / DETECTION_SCALE), 
           int(w / DETECTION_SCALE), int(h / DETECTION_SCALE))
        x, y, w, h = box

        return {
            KEY_found:      True,
            KEY_position:   (x, y),
            KEY_box:        (x, y, w, h),
            KEY_confidence: best_score,
            KEY_label:      best_label,
        }
    

class CharacterFinder:
    def __init__(self, sprite_ref_dir: str = "sprite_references") -> None:
        self.sprite_ref_dir = Path(sprite_ref_dir)
        self.characters = self._load_characters()
        print(f"[INFO] CharacterFinder loaded {len(self.characters)} character(s)")

    def _load_characters(self) -> dict:
        """
        Scans sprite_references/ and loads ONE sprite per character folder.
        Priority: loads 'default.png' if it exists, otherwise the first image found.
        Stores pre-computed edges to avoid recomputing on every detect() call.
        """
        characters = {}
        for char_dir in sorted(self.sprite_ref_dir.iterdir()):
            if not char_dir.is_dir():
                continue

            sprite_path = self._pick_sprite(char_dir)
            if sprite_path is None:
                print(f"[WARN] No valid images in {char_dir} — skipping")
                continue

            img = cv2.imread(str(sprite_path))
            if img is None:
                print(f"[WARN] Could not load {sprite_path} — skipping")
                continue

            characters[char_dir.name] = {
                "edges": to_edges(img),   # precomputed — never recomputed per frame
                "sprite": sprite_path.name,
            }
            print(f"[INFO]  {char_dir.name} <- {sprite_path.name}")

        return characters

    def _pick_sprite(self, char_dir: Path):
        """Returns default.png if present, otherwise the first image in the folder."""
        default = char_dir / "default.png"
        if default.exists():
            return default

        for path in sorted(char_dir.iterdir()):
            if path.suffix.lower() in allowedImageTypes:
                return path

        return None

    def MatchChar(self, frame_edges, data, char_name):
        box, conf, _ = multiscale_edge_match(frame_edges, data["edges"])
        if box and conf >= MATCH_THRESH:
            x, y, w, h = box
            return {
                "character": char_name,
                KEY_confidence: conf,
                KEY_position: (x, y),
                KEY_box: box,
            }
        return None  # explicit None when no match


    def find(self, frame: np.ndarray, top_n: int = 2) -> list[dict]:
        frame_edges = to_edges(frame)
        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.MatchChar, frame_edges=frame_edges, data=data, char_name=char_name): char_name
                for char_name, data in self.characters.items()
            }

            for future in as_completed(futures):
                char_name = futures[future]
                try:
                    result = future.result()
                    if result is not None:  # only add if it matched
                        results.append(result)
                except Exception as e:
                    print(f"{char_name} failed: {e}")

        results.sort(key=lambda r: r[KEY_confidence], reverse=True)
        return results[:top_n]