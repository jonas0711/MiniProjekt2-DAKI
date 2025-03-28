import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

# Konstanter
TILE_LABELS_PATH = "Excel+JSON/tile_labels_mapping.json"
EXTRACTED_TILES_PATH = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
OUTPUT_PATH = "HSV_CrownDetectionResults"

# HSV-intervaller for kroner per terræntype - justeret baseret på vores evaluering
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
    # Tjek om filen eksisterer
    if not os.path.exists(TILE_LABELS_PATH):
        print(f"Fejl: Kunne ikke finde JSON-filen {TILE_LABELS_PATH}")
        return {}
    
    try:
        with open(TILE_LABELS_PATH, 'r') as f:
            tile_labels = json.load(f)
        
        # Opret en mapping fra filnavn til terræntype og kroneantal
        filename_to_info = {}
        crown_count_by_terrain = defaultdict(int)
        total_tiles_by_terrain = defaultdict(int)
        
        for board_name, tiles in tile_labels.items():
            for tile_pos, tile_info in tiles.items():
                filename = tile_info["filename"]
                terrain = tile_info["terrain"]
                crowns = tile_info["crowns"]
                
                # Gem information med filnavnet
                filename_to_info[filename] = {
                    "terrain": terrain,
                    "crowns": crowns,
                    "board": board_name,
                    "position": tile_pos
                }
                
                # Opdater statistik
                if terrain not in ["Unknown", "Home", "Table"]:
                    total_tiles_by_terrain[terrain] += 1
                    if crowns > 0:
                        crown_count_by_terrain[terrain] += 1
        
        print(f"Indlæst information for {len(filename_to_info)} tiles fra JSON")
        print("\nFordeling af terræntyper:")
        for terrain, count in sorted(total_tiles_by_terrain.items()):
            crown_count = crown_count_by_terrain[terrain]
            crown_percentage = (crown_count / count) * 100 if count > 0 else 0
            print(f"  {terrain}: {count} tiles, {crown_count} med kroner ({crown_percentage:.1f}%)")
        
        return filename_to_info
    
    except Exception as e:
        print(f"Fejl ved indlæsning af JSON-filen: {e}")
        return {}

def create_outer_layer_mask(image, grid_size=3):
    """
    Opretter en maske for det yderste lag i en tile.
    
    Args:
        image: Billedet
        grid_size: Antal underopdelte celler (default: 3)
    
    Returns:
        numpy.ndarray: Maske med hvide pixels i det yderste lag
    """
    height, width = image.shape[:2]
    
    # Opretter en tom maske (sort)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Beregner størrelsen for hver under-tile
    sub_height = height // grid_size
    sub_width = width // grid_size
    
    # Sikrer at vi har gyldige sub-dimensioner
    if sub_height <= 0 or sub_width <= 0:
        # Fallback til simpel kant
        border_width = max(3, min(height, width) // 10)
        mask.fill(255)
        # Skær centrum ud
        center_y_start = border_width
        center_y_end = height - border_width
        center_x_start = border_width
        center_x_end = width - border_width
        
        if center_y_end > center_y_start and center_x_end > center_x_start:
            mask[center_y_start:center_y_end, center_x_start:center_x_end] = 0
        
        return mask
    
    # Fylder det yderste lag med hvid
    # Øverste og nederste række
    for i in range(grid_size):
        # Øverste række
        y_start = 0
        x_start = i * sub_width
        mask[y_start:y_start+sub_height, x_start:x_start+sub_width] = 255
        
        # Nederste række
        y_start = (grid_size-1) * sub_height
        mask[y_start:y_start+sub_height, x_start:x_start+sub_width] = 255
    
    # Venstre og højre kolonne (undtagen hjørnerne som allerede er dækket)
    for i in range(1, grid_size-1):
        # Venstre kolonne
        y_start = i * sub_height
        x_start = 0
        mask[y_start:y_start+sub_height, x_start:x_start+sub_width] = 255
        
        # Højre kolonne
        x_start = (grid_size-1) * sub_width
        mask[y_start:y_start+sub_height, x_start:x_start+sub_width] = 255
    
    return mask

def calculate_circularity(contour):
    """
    Beregner cirkularitet (4π × Area / Perimeter²) for en kontur.
    
    Args:
        contour: OpenCV kontur
        
    Returns:
        float: Cirkularitetsværdi mellem 0 og 1
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Undgå division med nul
    if perimeter == 0:
        return 0
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity

def calculate_aspect_ratio(contour):
    """
    Beregner aspect ratio (bredde/højde) for en kontur.
    
    Args:
        contour: OpenCV kontur
        
    Returns:
        float: Aspect ratio værdi
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    # Undgå division med nul
    if h == 0:
        return 0
    
    aspect_ratio = float(w) / h
    return aspect_ratio

def detect_crowns_hsv(image, terrain_type):
    """
    Detekterer kroner ved hjælp af HSV-farvefiltrering og formanalyse.
    
    Args:
        image: RGB-billede
        terrain_type: Terræntype (Field, Forest, Lake, osv.)
    
    Returns:
        tuple: (antal_kroner, kronerektangler, visualiseringsfigur)
    """
    # Lav en kopi af originalbilledet til visualisering
    original_image = image.copy()
    
    # 1. Konverter til HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 2. Anvend HSV-filtrering baseret på terræntype
    hsv_range = CROWN_HSV_RANGES.get(terrain_type, CROWN_HSV_RANGES['default'])
    lower_hsv = np.array(hsv_range[0])
    upper_hsv = np.array(hsv_range[1])
    
    # 3. Opret HSV-masken
    hsv_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    
    # 4. Opret yderlagsmasken (da kroner typisk er i yderområderne)
    outer_layer_mask = create_outer_layer_mask(image)
    
    # 5. Kombiner maskerne
    masked_result = cv2.bitwise_and(hsv_mask, outer_layer_mask)
    
    # 6. Anvend morfologiske operationer for at reducere støj
    kernel = np.ones((3, 3), np.uint8)
    # Åbning: fjern små objekter
    cleaned_mask = cv2.morphologyEx(masked_result, cv2.MORPH_OPEN, kernel, iterations=1)
    # Lukning: luk huller og forbind nærliggende områder
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 7. Find konturer
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 8. Filtrer konturer baseret på form og størrelse
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
    
    # 9. Fjern overlappende detektioner (non-max suppression)
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
    
    # 10. Antal kroner er antallet af filtrerede kandidater
    crown_count = len(filtered_candidates)
    
    # 11. Skab visualisering
    visualization = create_detection_visualization(
        original_image, hsv_image, hsv_mask, 
        outer_layer_mask, cleaned_mask, filtered_candidates, terrain_type
    )
    
    return crown_count, filtered_candidates, visualization

def create_detection_visualization(original_image, hsv_image, hsv_mask, 
                                  outer_layer_mask, cleaned_mask, 
                                  crown_candidates, terrain_type):
    """
    Skaber en visualisering af kronedetektionsprocessen.
    
    Args:
        original_image: Originalt RGB billede
        hsv_image: HSV-konverteret billede
        hsv_mask: HSV-baseret maske
        outer_layer_mask: Maske for det yderste lag
        cleaned_mask: Renset maske
        crown_candidates: Liste med detekterede kronekandidater
        terrain_type: Terræntypen (som string)
    
    Returns:
        matplotlib.figure.Figure: Matplotlib figur med visualisering
    """
    # Konverter HSV til RGB for visualisering
    hsv_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    # Opret figur med subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original billede
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # HSV-billede
    axes[0, 1].imshow(hsv_rgb)
    axes[0, 1].set_title('HSV')
    axes[0, 1].axis('off')
    
    # HSV-maske
    axes[0, 2].imshow(hsv_mask, cmap='gray')
    axes[0, 2].set_title('Hvid Maske (HSV)')
    axes[0, 2].axis('off')
    
    # Yderlagsmaske
    axes[1, 0].imshow(outer_layer_mask, cmap='gray')
    axes[1, 0].set_title('Yderlag Maske')
    axes[1, 0].axis('off')
    
    # Renset maske
    axes[1, 1].imshow(cleaned_mask, cmap='gray')
    axes[1, 1].set_title('Renset Maske')
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
    plt.suptitle(f'HSV-baseret Kronedetektion - {terrain_type}', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    return fig

def eval_crown_detection(ground_truth, detected_count):
    """
    Evaluerer kronedetektionen ved at sammenligne med ground truth.
    
    Args:
        ground_truth: Faktisk antal kroner
        detected_count: Detekteret antal kroner
        
    Returns:
        bool: True hvis detektionen er korrekt, False ellers
    """
    return ground_truth == detected_count

def process_test_set():
    """
    Behandler et testsæt af billeder og evaluerer kronedetektion.
    
    Returns:
        dict: Evalueringsresultater
    """
    # Indlæs tileinformation
    tile_info = load_tile_labels()
    if not tile_info:
        print("Ingen tile information indlæst, afslutter")
        return {}
    
    # Opret output-mappe hvis den ikke findes
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Find unikke terræntyper
    terrain_types = set()
    for _, info in tile_info.items():
        terrain = info["terrain"]
        if terrain not in ["Unknown", "Home", "Table"]:
            terrain_types.add(terrain)
    
    # Evalueringsresultater
    results = {}
    
    # Statistikvariable for samlet evaluering
    correct_total = 0
    total_samples = 0
    
    # Terrænspecifik statistik
    terrain_stats = {}
    
    # Behandl hver terræntype
    for terrain_type in sorted(terrain_types):
        print(f"\nBehandler terræntype: {terrain_type}")
        
        # Opret output-mappe for denne terræntype
        terrain_output_path = os.path.join(OUTPUT_PATH, terrain_type)
        if not os.path.exists(terrain_output_path):
            os.makedirs(terrain_output_path)
        
        # Find billeder for denne terræntype
        terrain_tiles = []
        for filename, info in tile_info.items():
            if info["terrain"] == terrain_type:
                terrain_tiles.append((filename, info))
        
        # Tællere for statistik
        correct_detections = 0
        total_detections = 0
        
        # Behandl op til 10 tilfældige billeder
        import random
        random.seed(42)  # For reproducerbarhed
        sample_tiles = random.sample(terrain_tiles, min(10, len(terrain_tiles)))
        
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
            
            print(f"  Behandler {filename} (Ground Truth: {true_crown_count} kroner)")
            
            # Detekter kroner
            detected_crown_count, crown_rects, visualization = detect_crowns_hsv(image, terrain_type)
            
            # Evaluer resultat
            correct = eval_crown_detection(true_crown_count, detected_crown_count)
            
            # Opdater statistik
            total_detections += 1
            if correct:
                correct_detections += 1
                result_text = "✓ KORREKT"
            else:
                result_text = "✗ FEJL"
            
            print(f"    - Detekteret {detected_crown_count} kroner - {result_text}")
            
            # Gem visualiseringen
            output_filename = f"{os.path.splitext(filename)[0]}_detection.png"
            output_path = os.path.join(terrain_output_path, output_filename)
            visualization.savefig(output_path)
            plt.close(visualization)
            
            # Gem resultatet
            results[filename] = {
                "terrain": terrain_type,
                "true_count": true_crown_count,
                "detected_count": detected_crown_count,
                "correct": correct
            }
        
        # Beregn terrænspecifik nøjagtighed
        accuracy = correct_detections / total_detections if total_detections > 0 else 0
        terrain_stats[terrain_type] = {
            'correct': correct_detections,
            'total': total_detections,
            'accuracy': accuracy
        }
        
        # Opdater samlet statistik
        correct_total += correct_detections
        total_samples += total_detections
        
        print(f"  Nøjagtighed for {terrain_type}: {accuracy:.2%} ({correct_detections}/{total_detections})")
    
    # Beregn samlet nøjagtighed
    overall_accuracy = correct_total / total_samples if total_samples > 0 else 0
    
    print("\n=== Samlet Evaluering ===")
    print(f"Samlet nøjagtighed: {overall_accuracy:.2%} ({correct_total}/{total_samples})")
    print("\nTerrænspecifik nøjagtighed:")
    for terrain, stats in sorted(terrain_stats.items()):
        print(f"  {terrain}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Tilføj statistik til resultater
    results['_statistics'] = {
        'overall_accuracy': overall_accuracy,
        'correct_total': correct_total,
        'total_samples': total_samples,
        'terrain_stats': terrain_stats
    }
    
    return results

def main():
    """
    Hovedfunktion til at køre tests af HSV-baseret kronedetektion.
    """
    print("=== Test af HSV-baseret Kronedetektion ===")
    
    # Kør tests
    results = process_test_set()
    
    print(f"\nResultater er gemt i {OUTPUT_PATH}")

if __name__ == "__main__":
    main()