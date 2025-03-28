import cv2
import numpy as np
import os

# Parametre
TEMPLATE_DIR = "KingDominoDataset/Crown_Templates"
MATCH_THRESHOLD = 0.63

# Indlæs alle templates i starten
templates = []
for filename in os.listdir(TEMPLATE_DIR):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(TEMPLATE_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append(img)

def detect_crowns_positions(tile, terrain_type=None):
    """
    Detekterer kroner vha. template matching i et tile.
    
    Args:
        tile: RGB-billede af én tile.
        terrain_type: (Ignoreres i denne version)

    Returns:
        crown_count: antal fundne kroner (maks 3).
        centroids: liste af (cx, cy) positioner i tile-koordinater.
    """
    gray_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    found_centers = []

    for template in templates:
        res = cv2.matchTemplate(gray_tile, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= MATCH_THRESHOLD)
        h, w = template.shape

        for pt in zip(*loc[::-1]):
            cx, cy = pt[0] + w // 2, pt[1] + h // 2
            found_centers.append((cx, cy))

    # Fjern overlappende detektioner (optionelt)
    filtered = []
    for (x, y) in found_centers:
        if all(np.hypot(x - fx, y - fy) > 10 for (fx, fy) in filtered):
            filtered.append((x, y))

    return min(len(filtered), 3), filtered[:3]
