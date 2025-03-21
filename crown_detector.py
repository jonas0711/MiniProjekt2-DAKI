import cv2
import numpy as np
from collections import Counter

def detect_crowns(tile, terrain_type):
    """
    Detekterer kroner i et tile baseret på terræntypen.
    
    Args:
        tile: RGB-billede af tile
        terrain_type: Type af terræn ('Field', 'Forest', etc.)
        
    Returns:
        int: Antal kroner detekteret
    """
    # Konverter til HSV (bedre til farvebaseret segmentering)
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    
    # Definér HSV-intervaller for kroner baseret på terræntype
    # Disse værdier skal justeres baseret på terrænets farve
    if terrain_type == 'Field':
        # For gult terræn (sværere at skelne) - strengere krav
        lower_gold = np.array([20, 150, 150])  # Mørkere gul/guld
        upper_gold = np.array([35, 255, 255])  # Lysere gul/guld
    else:
        # For andre terræntyper (lettere at skelne)
        lower_gold = np.array([15, 100, 100])  # Bredere interval
        upper_gold = np.array([40, 255, 255])
    
    # Opret maske baseret på farveinterval
    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    
    # Anvend morfologiske operationer for at fjerne støj
    kernel = np.ones((3, 3), np.uint8)
    # Opening (erosion efterfulgt af dilation) fjerner små støjområder
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Closing (dilation efterfulgt af erosion) lukker små huller i objekter
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find konturer i masken
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer konturer baseret på areal og form
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip små konturer (støj)
        if area < 50:
            continue
            
        # Beregn cirkularity (4π × Area / Perimeter²)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter baseret på cirkularity og areal
        # Kroner er typisk nogenlunde cirkulære og har en bestemt størrelse
        if circularity > 0.3 and area > 50 and area < 1000:
            filtered_contours.append(contour)
    
    # For Field (gult terræn), brug ekstra verifikation
    if terrain_type == 'Field' and len(filtered_contours) > 0:
        # For gult terræn: anvend kantdetektion som ekstra verifikation
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        verified_contours = []
        for contour in filtered_contours:
            # Opret en maske for konturen
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Tæl antal kantpixels inden for konturen
            edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
            
            # Hvis der er nok kanter inden for konturen, er det sandsynligvis en krone
            if edge_pixels > 10:
                verified_contours.append(contour)
        
        crown_count = len(verified_contours)
    else:
        crown_count = len(filtered_contours)
    
    # Sikrer at vi ikke overstiger 3 (det maksimale antal kroner på et felt i Kingdomino)
    return min(crown_count, 3)

def visualize_crown_detection(tile, terrain_type):
    """
    Visualiserer kronedetektionsprocessen for et tile.
    
    Args:
        tile: RGB-billede af tile
        terrain_type: Type af terræn ('Field', 'Forest', etc.)
        
    Returns:
        tuple: (originalt billede, maske, kontur-visualisering, antal kroner)
    """
    # Konverter til HSV
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    
    # Definér HSV-intervaller for kroner baseret på terræntype
    if terrain_type == 'Field':
        lower_gold = np.array([20, 150, 150])
        upper_gold = np.array([35, 255, 255])
    else:
        lower_gold = np.array([15, 100, 100])
        upper_gold = np.array([40, 255, 255])
    
    # Opret maske baseret på farveinterval
    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    
    # Anvend morfologiske operationer for at fjerne støj
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find konturer
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer konturer baseret på areal og form
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip små konturer (støj)
        if area < 50:
            continue
            
        # Beregn cirkularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter baseret på cirkularity og areal
        if circularity > 0.3 and area > 50 and area < 1000:
            filtered_contours.append(contour)
    
    # For Field (gult terræn), brug ekstra verifikation
    if terrain_type == 'Field' and len(filtered_contours) > 0:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        verified_contours = []
        for contour in filtered_contours:
            mask_contour = np.zeros_like(gray)
            cv2.drawContours(mask_contour, [contour], 0, 255, -1)
            edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask_contour))
            
            if edge_pixels > 10:
                verified_contours.append(contour)
        
        contours_to_use = verified_contours
        crown_count = len(verified_contours)
    else:
        contours_to_use = filtered_contours
        crown_count = len(filtered_contours)
    
    # Opret visualisering af konturer
    contour_vis = tile.copy()
    cv2.drawContours(contour_vis, contours_to_use, -1, (0, 255, 0), 2)
    
    # Tilføj tekst med antal kroner
    cv2.putText(contour_vis, f"Crowns: {crown_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return tile, mask, contour_vis, min(crown_count, 3)