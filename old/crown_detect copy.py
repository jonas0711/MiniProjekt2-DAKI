# Avanceret kronedetektion i Kingdomino-brikker
# Kombinerer HSV-segmentering, morfologi, kontur-analyse og Canny-kantdetektion

import cv2
import numpy as np

def detect_crowns(image_path, debug=False):
    # Indlæs billede
    image = cv2.imread(image_path)
    if image is None:
        print(f"Kunne ikke indlæse billedet: {image_path}")
        return []
        
    original = image.copy()
    # Større kernel for mere robust støjfjernelse
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Konverter til HSV og grå
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Juster farveområdet
    lower_gold = np.array([10, 80, 80])  # Lavere tærskel
    upper_gold = np.array([40, 255, 255])  # Højere tærskel
    mask_hsv = cv2.inRange(hsv, lower_gold, upper_gold)

    # Mere aggressiv morfologisk støjreduktion
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    # Dynamisk Canny-tærskel
    median_intensity = np.median(gray)
    lower_thresh = int(max(0, 0.7 * median_intensity))
    upper_thresh = int(min(255, 1.3 * median_intensity))
    edges = cv2.Canny(gray, lower_thresh, upper_thresh)

    # Find konturer på masken
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_images = {
            "Original": original,
            "Blurred": blurred,
            "HSV Mask": mask_hsv,
            "Clean Mask": mask_clean,
            "Edges": edges
        }
        for name, img in debug_images.items():
            cv2.imshow(name, img)

    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Justeret areal-interval baseret på typiske kroner
        if area < 25 or area > 1000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Mere afslappet cirkularity-threshold
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.2:
            continue

        # Bedre intern edge-tjek med proportion i stedet for absolut tal
        mask_cnt = np.zeros_like(gray)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, thickness=cv2.FILLED)
        inner_edges = cv2.bitwise_and(edges, edges, mask=mask_cnt)
        edge_count = cv2.countNonZero(inner_edges)
        edge_density = edge_count / area if area > 0 else 0
        if edge_density < 0.05:  # Mindst 5% af arealet skal være kanter
            continue

        # Centroid beregning
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        detected.append((cx, cy))
        cv2.circle(original, (cx, cy), 5, (0, 0, 255), -1)
        cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)

    print(f"Antal fundne kroner: {len(detected)}")
    for i, (x, y) in enumerate(detected):
        print(f"Krone {i+1}: ({x}, {y})")

    cv2.imshow("Detektion", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detected

if __name__ == "__main__":
    detect_crowns("22.jpg", debug=True)  # Erstat med ønsket billede, debug=True viser mellemtrin
