import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

# Konstanter
TILE_LABELS_PATH = "Excel+JSON/tile_labels_mapping.json"
EXTRACTED_TILES_PATH = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
OUTPUT_PATH = "CombinedCrownDetectionResults"

# Farvekonstant - specifik kronekant-farve (RGB værdi)
CROWN_OUTLINE_RGB = np.array([168, 171, 155])  # Den grå kantfarve vi så i billederne
CROWN_OUTLINE_TOLERANCE = 30  # Tolerance for farvevariation

# HSV-intervaller for kroner per terræntype - justeret baseret på billederne
CROWN_HSV_RANGES = {
    'Field': [(15, 40, 160), (40, 255, 255)],    # Guldtoner på gul baggrund
    'Forest': [(15, 50, 150), (40, 255, 255)],   # Guldtoner på grøn baggrund
    'Lake': [(15, 60, 150), (40, 255, 255)],     # Guldtoner på blå baggrund
    'Mine': [(15, 40, 140), (40, 255, 255)],     # Guldtoner på mørk baggrund
    'Swamp': [(15, 50, 150), (40, 255, 255)],    # Guldtoner på brun baggrund
    'Grassland': [(15, 50, 150), (40, 255, 255)],# Guldtoner på lysegrøn baggrund
    'default': [(15, 50, 150), (40, 255, 255)]   # Standardværdier
}

# Formfiltrering - parameterjustering per terræntype
SHAPE_PARAMS = {
    'Field': {'min_area': 20, 'max_area': 400, 'min_circularity': 0.3, 'max_aspect_ratio': 2.0},
    'Forest': {'min_area': 20, 'max_area': 400, 'min_circularity': 0.3, 'max_aspect_ratio': 2.0},
    'Lake': {'min_area': 20, 'max_area': 400, 'min_circularity': 0.2, 'max_aspect_ratio': 2.5},
    'Mine': {'min_area': 20, 'max_area': 400, 'min_circularity': 0.2, 'max_aspect_ratio': 2.5},
    'Swamp': {'min_area': 20, 'max_area': 400, 'min_circularity': 0.3, 'max_aspect_ratio': 2.0},
    'Grassland': {'min_area': 20, 'max_area': 400, 'min_circularity': 0.3, 'max_aspect_ratio': 2.0},
    'default': {'min_area': 20, 'max_area': 400, 'min_circularity': 0.3, 'max_aspect_ratio': 2.0}
}

def load_tile_labels():
    """
    Indlæser tile labels fra JSON-fil for at identificere kroner.
    
    Returns:
        dict: Mapping fra filnavn til information om terræntype og kroner
    """
    # Implementation er den samme som i HSV-versionen
    # ...

def create_outer_layer_mask(image, grid_size=3):
    """
    Opretter en maske for det yderste lag i en tile.
    
    Args:
        image: Billedet
        grid_size: Antal underopdelte celler (default: 3)
    
    Returns:
        numpy.ndarray: Maske med hvide pixels i det yderste lag
    """
    # Implementation er den samme som i HSV-versionen
    # ...

def calculate_circularity(contour):
    """
    Beregner cirkularitet (4π × Area / Perimeter²) for en kontur.
    
    Args:
        contour: OpenCV kontur
        
    Returns:
        float: Cirkularitetsværdi mellem 0 og 1
    """
    # Implementation er den samme som i HSV-versionen
    # ...

def calculate_aspect_ratio(contour):
    """
    Beregner aspect ratio (bredde/højde) for en kontur.
    
    Args:
        contour: OpenCV kontur
        
    Returns:
        float: Aspect ratio værdi
    """
    # Implementation er den samme som i HSV-versionen
    # ...

def create_specific_crown_color_mask(image, tolerance=CROWN_OUTLINE_TOLERANCE):
    """
    Skaber en maske for pixels, der matcher den specifikke kronekant-farve.
    
    Args:
        image: RGB billede
        tolerance: Farvetolerance (difference i RGB-kanaler)
        
    Returns:
        numpy.ndarray: Binær maske med hvide pixels, der matcher kronekant-farven
    """
    # Skab maske ved at se på afstand til den specifikke farve
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Beregn afstand i RGB-rum
    color_distance = np.sqrt(np.sum((image.astype(np.int32) - CROWN_OUTLINE_RGB) ** 2, axis=2))
    
    # Markér pixels der er inden for tolerancen
    mask[color_distance <= tolerance] = 255
    
    return mask

def detect_crowns_combined(image, terrain_type):
    """
    Detekterer kroner ved at kombinere HSV-farvefiltrering og specifik kantfarve.
    
    Args:
        image: RGB-billede
        terrain_type: Terræntype (Field, Forest, Lake, osv.)
    
    Returns:
        tuple: (antal_kroner, kronerektangler, visualiseringsfigur)
    """
    # Lav en kopi af originalbilledet til visualisering
    original_image = image.copy()
    
    # 1. Kontrastforbedring med CLAHE for bedre farvegenkendelse
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # 2. Opret en maske for den specifikke kronekant-farve
    specific_color_mask = create_specific_crown_color_mask(enhanced_image)
    
    # 3. Opret HSV-masken
    hsv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2HSV)
    hsv_range = CROWN_HSV_RANGES.get(terrain_type, CROWN_HSV_RANGES['default'])
    lower_hsv = np.array(hsv_range[0])
    upper_hsv = np.array(hsv_range[1])
    hsv_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    
    # 4. Kombiner maskerne 
    combined_mask = cv2.bitwise_or(specific_color_mask, hsv_mask)
    
    # 5. Opret yderlagsmasken
    outer_layer_mask = create_outer_layer_mask(image)
    
    # 6. Kombiner med yderlagsmasken
    masked_result = cv2.bitwise_and(combined_mask, outer_layer_mask)
    
    # 7. Anvend morfologiske operationer for at reducere støj
    kernel = np.ones((3, 3), np.uint8)
    # Åbning: fjern små objekter
    cleaned_mask = cv2.morphologyEx(masked_result, cv2.MORPH_OPEN, kernel, iterations=1)
    # Lukning: luk huller og forbind nærliggende områder
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 8. Find konturer
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 9. Filtrer konturer baseret på form og størrelse
    shape_params = SHAPE_PARAMS.get(terrain_type, SHAPE_PARAMS['default'])
    
    crown_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrer baseret på størrelse
        if shape_params['min_area'] <= area <= shape_params['max_area']:
            # Beregn cirkularitet
            circularity = calculate_circularity(contour)
            # Beregn aspect ratio
            aspect_ratio = calculate_aspect_ratio(contour)
            
            # Filtrer baseret på formparametre
            if (circularity >= shape_params['min_circularity'] and 
                aspect_ratio <= shape_params['max_aspect_ratio']):
                
                x, y, w, h = cv2.boundingRect(contour)
                crown_candidates.append((x, y, w, h, area, circularity, aspect_ratio))
    
    # 10. Fjern overlappende detektioner (non-max suppression)
    filtered_candidates = []
    for candidate in crown_candidates:
        x1, y1, w1, h1 = candidate[:4]
        
        # Tjek for overlap med allerede accepterede kandidater
        overlapping = False
        for accepted in filtered_candidates:
            x2, y2, w2, h2 = accepted[:4]
            
            # Beregn overlap
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            # Hvis overlap er stort, afvis kandidaten
            min_area = min(w1 * h1, w2 * h2)
            if overlap_area > 0.4 * min_area:  # 40% overlap threshold
                overlapping = True
                break
        
        if not overlapping:
            filtered_candidates.append(candidate)
    
    # 11. Antal kroner er antallet af filtrerede kandidater
    crown_count = len(filtered_candidates)
    
    # 12. Skab visualisering
    visualization = create_combined_visualization(
        original_image, enhanced_image, specific_color_mask, 
        hsv_mask, combined_mask, cleaned_mask, 
        filtered_candidates, terrain_type
    )
    
    return crown_count, filtered_candidates, visualization

def create_combined_visualization(original_image, enhanced_image, specific_color_mask, 
                                hsv_mask, combined_mask, cleaned_mask, 
                                crown_candidates, terrain_type):
    """
    Skaber en visualisering af den kombinerede kronedetektionsprocess.
    
    Args:
        original_image: Originalt RGB billede
        enhanced_image: Kontrastforbedret billede
        specific_color_mask: Maske for specifik kronekant-farve
        hsv_mask: HSV-baseret maske
        combined_mask: Kombineret maske
        cleaned_mask: Renset maske
        crown_candidates: Liste med detekterede kronekandidater
        terrain_type: Terræntypen (som string)
    
    Returns:
        matplotlib.figure.Figure: Matplotlib figur med visualisering
    """
    # Opret figur med subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original billede
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Kontrastforbedret billede
    axes[0, 1].imshow(enhanced_image)
    axes[0, 1].set_title('Kontrastforbedret')
    axes[0, 1].axis('off')
    
    # Specifik farve-maske
    axes[0, 2].imshow(specific_color_mask, cmap='gray')
    axes[0, 2].set_title(f'Specifik Kronekant Farve: {list(CROWN_OUTLINE_RGB)}')
    axes[0, 2].axis('off')
    
    # HSV-maske
    axes[1, 0].imshow(hsv_mask, cmap='gray')
    axes[1, 0].set_title('Hvid Maske (HSV)')
    axes[1, 0].axis('off')
    
    # Kombineret maske
    axes[1, 1].imshow(combined_mask, cmap='gray')
    axes[1, 1].set_title('Kombineret Maske')
    axes[1, 1].axis('off')
    
    # Resultat med detekterede kroner
    result_img = original_image.copy()
    for i, (x, y, w, h, area, circularity, aspect_ratio) in enumerate(crown_candidates):
        # Tegn rektangel
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Tilføj et nummer til hver detekteret krone
        cv2.putText(result_img, f"{i+1}", (x+w//2-5, y+h//2+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    axes[1, 2].imshow(result_img)
    axes[1, 2].set_title(f'{len(crown_candidates)} Kroner Detekteret')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Kombineret Kronedetektion - {terrain_type}', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    return fig

def compare_approaches():
    """
    Sammenligner HSV-baseret med kombineret tilgang på udvalgte billeder.
    
    Returns:
        dict: Sammenligningsresultater
    """
    # Indlæs tileinformation
    tile_info = load_tile_labels()
    if not tile_info:
        print("Ingen tile information indlæst, afslutter")
        return {}
    
    # Opret output-mappe hvis den ikke findes
    output_path = "ComparisonResults"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Find unikke terræntyper
    terrain_types = set()
    for _, info in tile_info.items():
        terrain = info["terrain"]
        if terrain not in ["Unknown", "Home", "Table"]:
            terrain_types.add(terrain)
    
    # Sammenligningsresultater
    comparison_results = {}
    
    # Statistikvariable
    hsv_correct = 0
    combined_correct = 0
    total_samples = 0
    
    # Behandl hver terræntype
    for terrain_type in sorted(terrain_types):
        print(f"\nSammenligner metoder for terræntype: {terrain_type}")
        
        # Find billeder for denne terræntype med kroner
        terrain_tiles = []
        for filename, info in tile_info.items():
            if info["terrain"] == terrain_type and info["crowns"] > 0:
                terrain_tiles.append((filename, info))
        
        # Vælg 2 tilfældige billeder
        import random
        random.seed(42)  # For reproducerbarhed
        sample_tiles = random.sample(terrain_tiles, min(2, len(terrain_tiles)))
        
        for filename, info in sample_tiles:
            # Indlæs billedet
            image_path = os.path.join(EXTRACTED_TILES_PATH, filename)
            
            if not os.path.exists(image_path):
                print(f"  Advarsel: Kunne ikke finde {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Advarsel: Kunne ikke indlæse {image_path}")
                continue
            
            # Konverter fra BGR til RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ground truth kroneantal
            true_crown_count = info["crowns"]
            
            print(f"  Sammenligner metoder for {filename} (Ground Truth: {true_crown_count} kroner)")
            
            # Kør HSV-baseret detektering
            hsv_count, _, hsv_vis = detect_crowns_hsv(image, terrain_type)
            
            # Kør kombineret detektering
            combined_count, _, combined_vis = detect_crowns_combined(image, terrain_type)
            
            # Evaluer resultater
            hsv_correct_detect = (hsv_count == true_crown_count)
            combined_correct_detect = (combined_count == true_crown_count)
            
            # Opdater statistik
            total_samples += 1
            if hsv_correct_detect:
                hsv_correct += 1
            if combined_correct_detect:
                combined_correct += 1
            
            print(f"    - HSV-baseret: {hsv_count} kroner - {'✓' if hsv_correct_detect else '✗'}")
            print(f"    - Kombineret: {combined_count} kroner - {'✓' if combined_correct_detect else '✗'}")
            
            # Gem visualiseringer
            hsv_vis.savefig(os.path.join(output_path, f"{filename.split('.')[0]}_hsv.png"))
            combined_vis.savefig(os.path.join(output_path, f"{filename.split('.')[0]}_combined.png"))
            plt.close(hsv_vis)
            plt.close(combined_vis)
            
            # Gem sammenligningsresultater
            comparison_results[filename] = {
                "terrain": terrain_type,
                "true_count": true_crown_count,
                "hsv_count": hsv_count,
                "hsv_correct": hsv_correct_detect,
                "combined_count": combined_count,
                "combined_correct": combined_correct_detect
            }
    
    # Beregn samlet statistik
    hsv_accuracy = hsv_correct / total_samples if total_samples > 0 else 0
    combined_accuracy = combined_correct / total_samples if total_samples > 0 else 0
    
    print("\n=== Samlet Sammenligning ===")
    print(f"HSV-baseret nøjagtighed: {hsv_accuracy:.2%} ({hsv_correct}/{total_samples})")
    print(f"Kombineret nøjagtighed: {combined_accuracy:.2%} ({combined_correct}/{total_samples})")
    
    # Tilføj statistik til resultater
    comparison_results['_statistics'] = {
        'hsv_accuracy': hsv_accuracy,
        'combined_accuracy': combined_accuracy,
        'hsv_correct': hsv_correct,
        'combined_correct': combined_correct,
        'total_samples': total_samples
    }
    
    return comparison_results

def main():
    """
    Hovedfunktion til at køre sammenligning af metoderne.
    """
    print("=== Sammenligning af HSV og Kombineret Kronedetektion ===")
    
    # Kør sammenligning
    results = compare_approaches()
    
    # Hvis den kombinerede metode er bedre
    if results['_statistics']['combined_accuracy'] > results['_statistics']['hsv_accuracy']:
        print("Den kombinerede metode opnår bedre resultater!")
        # Her kunne vi køre en fuld test af den kombinerede metode
    else:
        print("HSV-baseret metode er tilstrækkelig!")
        # Her kunne vi køre en fuld test af HSV-metoden

if __name__ == "__main__":
    main()