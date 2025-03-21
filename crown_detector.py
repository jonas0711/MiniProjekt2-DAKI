import cv2
import numpy as np

# Justerede parametre (se tidligere versioner)
HSV_THRESHOLDS = {
    'Field': {
        'lower': np.array([20, 150, 150]),
        'upper': np.array([30, 255, 255])
    },
    'Default': {
        'lower': np.array([18, 120, 120]),
        'upper': np.array([30, 255, 255])
    }
}

MIN_AREA = 60
MAX_AREA = 800
MIN_CIRCULARITY = 0.35
EDGE_THRESHOLD1 = 100
EDGE_THRESHOLD2 = 220
MIN_EDGE_PIXELS = 10
GAUSSIAN_KERNEL = (3, 3)

def detect_crowns_positions(tile, terrain_type):
    """
    Detekterer kroner i et tile og returnerer både antallet og positioner (centroider).
    
    Args:
        tile: RGB-billede af tile.
        terrain_type: Terræntype (f.eks. 'Field', 'Forest', etc.)
    
    Returns:
        crown_count: Antal kroner (maks 3)
        centroids: Liste af (cx, cy) centroid-koordinater relativt til tile.
    """
    blurred = cv2.GaussianBlur(tile, GAUSSIAN_KERNEL, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    thresholds = HSV_THRESHOLDS.get(terrain_type, HSV_THRESHOLDS['Default'])
    lower_gold = thresholds['lower']
    upper_gold = thresholds['upper']
    
    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        filtered_contours.append(contour)
    
    # Ekstra verifikation for 'Field'
    if terrain_type == 'Field' and filtered_contours:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
        verified_contours = []
        for contour in filtered_contours:
            c_mask = np.zeros_like(gray)
            cv2.drawContours(c_mask, [contour], -1, 255, -1)
            edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=c_mask))
            if edge_pixels > MIN_EDGE_PIXELS:
                verified_contours.append(contour)
        filtered_contours = verified_contours

    centroids = []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    
    crown_count = min(len(centroids), 3)
    # Returner kun de første crown_count positioner
    return crown_count, centroids[:crown_count]

# Den gamle detect_crowns() kan nu blot kaldes hvis man kun ønsker antallet
def detect_crowns(tile, terrain_type):
    count, _ = detect_crowns_positions(tile, terrain_type)
    return count
