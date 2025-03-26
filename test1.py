import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re

# Konstanter
TILE_LABELS_PATH = "Excel+JSON/tile_labels_mapping.json"
EXTRACTED_TILES_PATH = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
OUTPUT_PATH = "CrownDetectionResults"

# HSV-intervaller for guld/kroner - tilpasset for forskellige terræntyper
TERRAIN_HSV_RANGES = {
    'Field': [(15, 40, 100), (35, 255, 255)],   # Mætningskrav reduceret for Field
    'Forest': [(15, 30, 100), (35, 255, 255)],  # Lavere mætningskrav for Forest
    'Lake': [(15, 50, 100), (35, 255, 255)],    # Standard "Lavere Mætning" for Lake
    'Mine': [(10, 30, 100), (40, 255, 255)],    # Endnu lavere mætning og bredere interval for Mine
    'Swamp': [(15, 40, 100), (35, 255, 255)],   # Tilpasset for Swamp
    'Grassland': [(15, 50, 100), (35, 255, 255)], # Standard "Lavere Mætning" for Grassland
    'default': [(15, 50, 100), (35, 255, 255)]  # Standardværdier for ukendte terræntyper
}

# Morfologiske filterparametre for forskellige terræntyper
MORPH_PARAMS = {
    'Field': {'min_area': 30, 'max_area': 300, 'min_circularity': 0.5, 'max_aspect_ratio': 1.5},
    'Forest': {'min_area': 30, 'max_area': 300, 'min_circularity': 0.6, 'max_aspect_ratio': 1.5},  # Højere circularity krav
    'Lake': {'min_area': 30, 'max_area': 300, 'min_circularity': 0.4, 'max_aspect_ratio': 1.8},    # Mere fleksibelt for Lake
    'Mine': {'min_area': 30, 'max_area': 400, 'min_circularity': 0.4, 'max_aspect_ratio': 2.0},    # Mere fleksibelt for Mine
    'Swamp': {'min_area': 30, 'max_area': 300, 'min_circularity': 0.5, 'max_aspect_ratio': 1.5},
    'Grassland': {'min_area': 30, 'max_area': 300, 'min_circularity': 0.5, 'max_aspect_ratio': 1.5},
    'default': {'min_area': 30, 'max_area': 300, 'min_circularity': 0.5, 'max_aspect_ratio': 1.5}
}

def load_tile_labels():
    """Indlæser tile labels for at identificere kroner"""
    # Tjek om filen eksisterer
    if not os.path.exists(TILE_LABELS_PATH):
        print(f"Fejl: Kunne ikke finde JSON-filen {TILE_LABELS_PATH}")
        return {}
    
    try:
        with open(TILE_LABELS_PATH, 'r') as f:
            tile_labels = json.load(f)
        
        # Opret en mapping fra filnavn til terræntype og kroneantal
        filename_to_info = {}
        crown_count_by_terrain = {}
        
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
                
                # Tæl efter terræntype
                if terrain not in crown_count_by_terrain:
                    crown_count_by_terrain[terrain] = 0
                if crowns > 0:
                    crown_count_by_terrain[terrain] += 1
        
        print(f"Indlæst information for {len(filename_to_info)} tiles fra JSON")
        print("\nKronetal fordelt på terræntype:")
        for terrain, count in crown_count_by_terrain.items():
            print(f"  {terrain}: {count} tiles med kroner")
        
        return filename_to_info
    
    except Exception as e:
        print(f"Fejl ved indlæsning af JSON-filen: {e}")
        return {}

def create_outer_layer_mask(image, grid_size=5):
    """Opretter en maske for det yderste lag i en grid-opdelt tile"""
    height, width = image.shape[:2]
    
    # Opretter en tom maske (sort)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Beregner størrelsen for hver under-tile
    sub_height = height // grid_size
    sub_width = width // grid_size
    
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
    """Beregner cirkularitet (4π × Area / Perimeter²) for en kontur"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Undgå division med nul
    if perimeter == 0:
        return 0
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity

def calculate_aspect_ratio(contour):
    """Beregner aspect ratio (bredde/højde) for en kontur"""
    x, y, w, h = cv2.boundingRect(contour)
    
    # Undgå division med nul
    if h == 0:
        return 0
    
    aspect_ratio = float(w) / h
    return aspect_ratio

def detect_crowns(image, terrain_type):
    """
    Detekterer kroner i et billede baseret på terræntype
    
    Args:
        image: RGB-billede
        terrain_type: Terræntype (Field, Forest, Lake, osv.)
    
    Returns:
        tuple: (antal_kroner, kronerektangler, visualiseringsfigur)
    """
    # Konverter til HSV (bedre farvedetektion)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Hent terrænspecifikke HSV-intervaller
    hsv_range = TERRAIN_HSV_RANGES.get(terrain_type, TERRAIN_HSV_RANGES['default'])
    lower_gold = np.array(hsv_range[0])
    upper_gold = np.array(hsv_range[1])
    
    # Opret maske for guldområder
    gold_mask = cv2.inRange(hsv_image, lower_gold, upper_gold)
    
    # Opret maske for det yderste lag
    outer_layer_mask = create_outer_layer_mask(image)
    
    # Kombiner maskerne (begrænser søgningen til det yderste lag)
    combined_mask = cv2.bitwise_and(gold_mask, outer_layer_mask)
    
    # Anvend morfologiske operationer for at reducere støj
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find konturer af potentielle kroneområder
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Hent morfologiske filterparametre for terræntypen
    morph_params = MORPH_PARAMS.get(terrain_type, MORPH_PARAMS['default'])
    min_area = morph_params['min_area']
    max_area = morph_params['max_area']
    min_circularity = morph_params['min_circularity']
    max_aspect_ratio = morph_params['max_aspect_ratio']
    
    # Filtrer konturer baseret på area, circularity og aspect ratio
    crown_rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrer baseret på størrelse
        if min_area <= area <= max_area:
            # Beregn circularity
            circularity = calculate_circularity(contour)
            # Beregn aspect ratio
            aspect_ratio = calculate_aspect_ratio(contour)
            
            # Filtrer baseret på circularity og aspect ratio
            if circularity >= min_circularity and aspect_ratio <= max_aspect_ratio:
                x, y, w, h = cv2.boundingRect(contour)
                crown_rects.append((x, y, w, h, area, circularity, aspect_ratio))
    
    # Anvend non-max suppression for at undgå overlappende detektioner
    filtered_rects = []
    if crown_rects:
        # Sortér efter størrelse (største først)
        crown_rects.sort(key=lambda r: r[4], reverse=True)
        
        for rect in crown_rects:
            x1, y1, w1, h1 = rect[:4]
            overlapping = False
            
            # Tjek for overlap med allerede accepterede rektangler
            for filtered_rect in filtered_rects:
                x2, y2, w2, h2 = filtered_rect[:4]
                
                # Beregn overlap
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                # Hvis overlap er stort nok, afvis denne
                min_area = min(w1 * h1, w2 * h2)
                if overlap_area > 0.5 * min_area:
                    overlapping = True
                    break
            
            if not overlapping:
                filtered_rects.append(rect)
    
    # Antal kroner er antal godkendte rektangler
    crown_count = len(filtered_rects)
    
    # Skab en visualisering
    visualization = create_detection_visualization(
        image, hsv_image, gold_mask, outer_layer_mask, 
        cleaned_mask, filtered_rects, terrain_type
    )
    
    return crown_count, filtered_rects, visualization

def create_detection_visualization(image, hsv_image, gold_mask, outer_mask, 
                                   cleaned_mask, crown_rects, terrain_type):
    """
    Skaber en visuel fremstilling af kronedetektion for evalueringsformål
    
    Args:
        image: Originalt RGB billede
        hsv_image: HSV-konverteret billede
        gold_mask: Maske for guldfarve
        outer_mask: Maske for yderste lag
        cleaned_mask: Kombineret og renset maske
        crown_rects: Liste af detekterede kronerektangler (x, y, w, h, area, circularity, aspect_ratio)
        terrain_type: Terræn-type strengen
    
    Returns:
        matplotlib.figure.Figure: Matplotlib figur med visualisering
    """
    # Konverter HSV tilbage til RGB for visualisering
    hsv_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    # Opret figur med subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original billede
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # HSV repræsentation
    axes[0, 1].imshow(hsv_rgb)
    axes[0, 1].set_title('HSV Repræsentation')
    axes[0, 1].axis('off')
    
    # Guld maske
    axes[0, 2].imshow(gold_mask, cmap='gray')
    axes[0, 2].set_title('Guld Maske')
    axes[0, 2].axis('off')
    
    # Yderlagsmaske
    axes[1, 0].imshow(outer_mask, cmap='gray')
    axes[1, 0].set_title('Yderlag Maske')
    axes[1, 0].axis('off')
    
    # Renset maske
    axes[1, 1].imshow(cleaned_mask, cmap='gray')
    axes[1, 1].set_title('Renset Maske')
    axes[1, 1].axis('off')
    
    # Resultat med markerede kroner
    result_img = image.copy()
    for rect in crown_rects:
        x, y, w, h, area, circularity, aspect_ratio = rect
        # Tegn rektangel
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    axes[1, 2].imshow(result_img)
    axes[1, 2].set_title(f'{len(crown_rects)} Kroner Detekteret')
    axes[1, 2].axis('off')
    
    # Tilføj detaljetekst under hvert krone-rektangel
    for i, rect in enumerate(crown_rects):
        x, y, w, h, area, circularity, aspect_ratio = rect
        # Beregn position (midten af rektanglet)
        text_x = x + w/2
        text_y = y + h/2
        axes[1, 2].text(text_x, text_y, f"{i+1}", color='white', 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(facecolor='green', alpha=0.5, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    plt.suptitle(f'Kronedetektion - {terrain_type}', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    return fig

def process_tile_images_by_terrain():
    """
    Behandler alle tiles sorteret efter terræntype, detekterer kroner,
    og evaluerer resultaterne
    """
    # Indlæs tileinformation
    tile_info = load_tile_labels()
    if not tile_info:
        print("Ingen tile information indlæst, afslutter")
        return
    
    # Opret output-mappe hvis den ikke findes
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Opret en dictionary til at gemme tiles efter terræntype
    tiles_by_terrain = {}
    
    # Organiser tiles efter terræntype
    for filename, info in tile_info.items():
        terrain = info["terrain"]
        
        # Spring specielle terræntyper over
        if terrain in ["Unknown", "Home", "Table"]:
            continue
        
        if terrain not in tiles_by_terrain:
            tiles_by_terrain[terrain] = []
        
        # Tilføj filnavn og information
        tiles_by_terrain[terrain].append((filename, info))
    
    # Detektionsresultater
    results = {}
    
    # Behandl hver terræntype
    for terrain_type, tiles in tiles_by_terrain.items():
        print(f"\nBehandler {len(tiles)} tiles af terræntypen: {terrain_type}")
        
        # Opret output-mappe for denne terræntype
        terrain_output_path = os.path.join(OUTPUT_PATH, terrain_type)
        if not os.path.exists(terrain_output_path):
            os.makedirs(terrain_output_path)
        
        # Tæller for statistik
        correct_detections = 0
        total_detections = 0
        
        # Behandl op til 20 billeder per terræntype (for at spare tid)
        for i, (filename, info) in enumerate(tiles[:20]):
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
            detected_crown_count, crown_rects, visualization = detect_crowns(image, terrain_type)
            
            # Gem resultatet
            results[filename] = {
                "terrain": terrain_type,
                "true_count": true_crown_count,
                "detected_count": detected_crown_count,
                "crown_rects": crown_rects
            }
            
            # Opdater statistik
            total_detections += 1
            if detected_crown_count == true_crown_count:
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
        
        # Vis terrænspecifik statistik
        if total_detections > 0:
            accuracy = correct_detections / total_detections
            print(f"  Nøjagtighed for {terrain_type}: {correct_detections}/{total_detections} ({accuracy:.2%})")
    
    # Beregn samlet statistik
    total_correct = sum(1 for filename, res in results.items() 
                       if res["detected_count"] == res["true_count"])
    total_count = len(results)
    
    if total_count > 0:
        overall_accuracy = total_correct / total_count
        print(f"\nSamlet nøjagtighed: {total_correct}/{total_count} ({overall_accuracy:.2%})")
    
    return results

def main():
    """Hovedfunktion til at køre kronedetektion"""
    print("=== Kingdomino Kronedetektion med Morfologisk Filtrering ===")
    
    # Behandl tiles efter terræntype
    results = process_tile_images_by_terrain()
    
    print("\nKronedetektion afsluttet. Detaljerede resultater er gemt i:", OUTPUT_PATH)

if __name__ == "__main__":
    main()