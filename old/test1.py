import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
from collections import defaultdict
import random

# Konstanter
TILE_LABELS_PATH = "Excel+JSON/tile_labels_mapping.json"
EXTRACTED_TILES_PATH = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
OUTPUT_PATH = "CrownDetectionResults"

# Forbedrede HSV-intervaller for guld/kroner baseret på evalueringen
TERRAIN_HSV_RANGES = {
    'Field': [(15, 35, 100), (35, 255, 255)],    # Justeret mætning
    'Forest': [(15, 25, 100), (40, 255, 255)],   # Bredere interval for Forest
    'Lake': [(15, 40, 100), (40, 255, 255)],     # Bredere interval for Lake
    'Mine': [(10, 25, 100), (45, 255, 255)],     # Endnu bredere for Mine
    'Swamp': [(15, 35, 100), (40, 255, 255)],    # Justeret for Swamp
    'Grassland': [(15, 40, 100), (40, 255, 255)],# Bredere interval for Grassland
    'default': [(15, 40, 100), (40, 255, 255)]   # Standardværdier
}

# Opdaterede morfologiske filterparametre - mindre restriktive
MORPH_PARAMS = {
    'Field': {'min_area': 25, 'max_area': 350, 'min_circularity': 0.4, 'max_aspect_ratio': 1.8},
    'Forest': {'min_area': 25, 'max_area': 350, 'min_circularity': 0.4, 'max_aspect_ratio': 1.8},
    'Lake': {'min_area': 25, 'max_area': 350, 'min_circularity': 0.3, 'max_aspect_ratio': 2.0},  # Meget lavere circularity krav
    'Mine': {'min_area': 25, 'max_area': 450, 'min_circularity': 0.3, 'max_aspect_ratio': 2.2},  # Endnu mere fleksibelt
    'Swamp': {'min_area': 25, 'max_area': 350, 'min_circularity': 0.4, 'max_aspect_ratio': 1.8},
    'Grassland': {'min_area': 25, 'max_area': 350, 'min_circularity': 0.4, 'max_aspect_ratio': 1.8},
    'default': {'min_area': 25, 'max_area': 350, 'min_circularity': 0.4, 'max_aspect_ratio': 1.8}
}

# Morfologiske parametre for rensning af masker - mindre aggressiv
CLEANING_PARAMS = {
    'Field': {'open_iterations': 1, 'close_iterations': 1},
    'Forest': {'open_iterations': 1, 'close_iterations': 2},  # Mere closing for Forest
    'Lake': {'open_iterations': 0, 'close_iterations': 2},    # Ingen opening for Lake
    'Mine': {'open_iterations': 0, 'close_iterations': 2},    # Ingen opening for Mine
    'Swamp': {'open_iterations': 1, 'close_iterations': 1},
    'Grassland': {'open_iterations': 1, 'close_iterations': 1},
    'default': {'open_iterations': 1, 'close_iterations': 1}
}

def load_tile_labels():
    """
    Indlæser tile labels for at identificere kroner
    
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
    Opretter en maske for det yderste lag i en tile
    
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
        border_width = max(3, min(height, width) // 10)  # Brug mindst 3 pixels, højst 10% af billedstørrelse
        # Fyld hele billedet
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
    
    # Kontroller at masken faktisk har hvide pixels
    if cv2.countNonZero(mask) < 10:  # Hvis der er færre end 10 hvide pixels
        # Fallback til en enkel kantmaske
        border_width = max(3, min(height, width) // 10)
        mask.fill(255)
        # Skær centrum ud
        mask[border_width:height-border_width, border_width:width-border_width] = 0
    
    return mask

def calculate_circularity(contour):
    """
    Beregner cirkularitet (4π × Area / Perimeter²) for en kontur
    
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
    Beregner aspect ratio (bredde/højde) for en kontur
    
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

def calculate_confidence_score(contour, terrain_type):
    """
    Beregner en konfidensscore for en krone-kandidat
    
    Args:
        contour: OpenCV kontur
        terrain_type: Terræntype
        
    Returns:
        float: Konfidensscore mellem 0 og 1
    """
    # Beregn grundlæggende metrikker
    area = cv2.contourArea(contour)
    circularity = calculate_circularity(contour)
    aspect_ratio = calculate_aspect_ratio(contour)
    
    # Hent ideelle parametre for denne terræntype
    params = MORPH_PARAMS.get(terrain_type, MORPH_PARAMS['default'])
    
    # Beregn scores for hver metrik (0-1)
    area_score = 0
    if params['min_area'] <= area <= params['max_area']:
        # Ideelt område er midten af min og max
        ideal_area = (params['min_area'] + params['max_area']) / 2
        # Normaliseret afstand til ideal (0 = ideal, 1 = grænser)
        normalized_distance = abs(area - ideal_area) / (params['max_area'] - params['min_area']) * 2
        # Invertér så 1 = ideal, 0 = grænser
        area_score = 1.0 - min(1.0, normalized_distance)
    
    # Circularity score (højere er bedre, men med vægtet betydning pr. terræntype)
    circularity_score = min(1.0, circularity / 0.8)  # 0.8 betragtes som perfekt cirkel for dette formål
    
    # Aspect ratio score (1.0 = kvadratisk, mindre for ikke-kvadratiske)
    aspect_score = min(1.0, 1.0 / aspect_ratio if aspect_ratio > 1.0 else aspect_ratio)
    
    # Vægt faktorerne forskelligt baseret på terræntype
    if terrain_type == 'Lake' or terrain_type == 'Mine':
        # For Lake og Mine er circularity mindre vigtigt
        weights = {'area': 0.5, 'circularity': 0.2, 'aspect': 0.3}
    else:
        # For andre terræntyper er circularity vigtigere
        weights = {'area': 0.4, 'circularity': 0.4, 'aspect': 0.2}
    
    # Beregn vægtet gennemsnit
    confidence = (
        area_score * weights['area'] +
        circularity_score * weights['circularity'] +
        aspect_score * weights['aspect']
    )
    
    return confidence

def detect_crowns(image, terrain_type):
    """
    Detekterer kroner i et billede baseret på terræntype med forbedret algoritme
    
    Args:
        image: RGB-billede
        terrain_type: Terræntype (Field, Forest, Lake, osv.)
    
    Returns:
        tuple: (antal_kroner, kronerektangler, visualiseringsfigur, konfidenser)
    """
    # Konverter til HSV (bedre farvedetektion)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Hent terrænspecifikke HSV-intervaller
    hsv_range = TERRAIN_HSV_RANGES.get(terrain_type, TERRAIN_HSV_RANGES['default'])
    lower_gold = np.array(hsv_range[0])
    upper_gold = np.array(hsv_range[1])
    
    # Opret maske for guldområder
    gold_mask = cv2.inRange(hsv_image, lower_gold, upper_gold)
    
    # Opret maske for det yderste lag med forbedret metode
    # Prøv med både 3x3 og 5x5 grid, brug den der giver bedst resultat
    outer_layer_mask_3 = create_outer_layer_mask(image, grid_size=3)
    outer_layer_mask_5 = create_outer_layer_mask(image, grid_size=5)
    
    # Vælg masken med flest hvide pixels
    if cv2.countNonZero(outer_layer_mask_5) > cv2.countNonZero(outer_layer_mask_3):
        outer_layer_mask = outer_layer_mask_5
    else:
        outer_layer_mask = outer_layer_mask_3
    
    # Kombiner maskerne (begrænser søgningen til det yderste lag)
    combined_mask = cv2.bitwise_and(gold_mask, outer_layer_mask)
    
    # Hent terrænspecifikke rensningsparametre
    cleaning = CLEANING_PARAMS.get(terrain_type, CLEANING_PARAMS['default'])
    
    # Anvend morfologiske operationer for at reducere støj - mindre aggressivt
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = combined_mask.copy()
    
    # Kun anvend opening hvis det er specificeret
    if cleaning['open_iterations'] > 0:
        cleaned_mask = cv2.morphologyEx(
            cleaned_mask, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=cleaning['open_iterations']
        )
    
    # Anvend closing med terrænspecifikt antal iterationer
    cleaned_mask = cv2.morphologyEx(
        cleaned_mask, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=cleaning['close_iterations']
    )
    
    # Find konturer af potentielle kroneområder
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Hent morfologiske filterparametre for terræntypen
    morph_params = MORPH_PARAMS.get(terrain_type, MORPH_PARAMS['default'])
    min_area = morph_params['min_area']
    max_area = morph_params['max_area']
    min_circularity = morph_params['min_circularity']
    max_aspect_ratio = morph_params['max_aspect_ratio']
    
    # Filtrer konturer baseret på area, circularity og aspect ratio
    # samt beregn en konfidensscore for hver kandidat
    crown_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrer baseret på størrelse
        if min_area <= area <= max_area:
            # Beregn circularity
            circularity = calculate_circularity(contour)
            # Beregn aspect ratio
            aspect_ratio = calculate_aspect_ratio(contour)
            
            # Filtrer baseret på circularity og aspect ratio - men med mere fleksibilitet
            if circularity >= min_circularity * 0.8 and aspect_ratio <= max_aspect_ratio * 1.2:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Beregn konfidensscore
                confidence = calculate_confidence_score(contour, terrain_type)
                
                crown_candidates.append({
                    'rect': (x, y, w, h),
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence,
                    'contour': contour
                })
    
    # Sortér kandidater efter konfidens (højeste først)
    crown_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Anvend non-max suppression for at undgå overlappende detektioner
    filtered_candidates = []
    
    # Brug dynamisk threshold baseret på terræntype
    confidence_threshold = 0.5
    if terrain_type == 'Lake' or terrain_type == 'Mine':
        confidence_threshold = 0.4  # Lavere threshold for udfordrende terræntyper
    
    for candidate in crown_candidates:
        # Acceptér kun kandidater over konfidenstærsklen
        if candidate['confidence'] < confidence_threshold:
            continue
            
        x1, y1, w1, h1 = candidate['rect']
        overlapping = False
        
        # Tjek for overlap med allerede accepterede kandidater
        for accepted in filtered_candidates:
            x2, y2, w2, h2 = accepted['rect']
            
            # Beregn overlap
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            # Hvis overlap er stort nok, afvis denne
            min_area = min(w1 * h1, w2 * h2)
            if overlap_area > 0.3 * min_area:  # Reduceret overlap tærskel
                # Hvis den nuværende kandidat har højere konfidens, udskift den accepterede
                if candidate['confidence'] > accepted['confidence'] + 0.2:  # Signifikant højere
                    filtered_candidates.remove(accepted)
                    overlapping = False
                    break
                else:
                    overlapping = True
                    break
        
        if not overlapping:
            filtered_candidates.append(candidate)
    
    # Konverter tilbage til det format resten af koden forventer
    filtered_rects = [c['rect'] + (c['area'], c['circularity'], c['aspect_ratio']) for c in filtered_candidates]
    confidences = [c['confidence'] for c in filtered_candidates]
    
    # Antal kroner er antal godkendte kandidater
    crown_count = len(filtered_candidates)
    
    # Skab en visualisering
    visualization = create_detection_visualization(
        image, hsv_image, gold_mask, outer_layer_mask, 
        cleaned_mask, filtered_candidates, terrain_type
    )
    
    return crown_count, filtered_rects, visualization, confidences

def create_detection_visualization(image, hsv_image, gold_mask, outer_mask, 
                                   cleaned_mask, crown_candidates, terrain_type):
    """
    Skaber en visuel fremstilling af kronedetektion for evalueringsformål
    
    Args:
        image: Originalt RGB billede
        hsv_image: HSV-konverteret billede
        gold_mask: Maske for guldfarve
        outer_mask: Maske for yderste lag
        cleaned_mask: Kombineret og renset maske
        crown_candidates: Liste af detekterede kronekandidater
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
    for candidate in crown_candidates:
        x, y, w, h = candidate['rect']
        confidence = candidate['confidence']
        
        # Farv rektanglet baseret på konfidens (grøn = høj, gul = medium, rød = lav)
        if confidence >= 0.7:
            color = (0, 255, 0)  # Grøn
        elif confidence >= 0.5:
            color = (255, 255, 0)  # Gul
        else:
            color = (255, 0, 0)  # Rød
            
        # Tegn rektangel
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
    axes[1, 2].imshow(result_img)
    axes[1, 2].set_title(f'{len(crown_candidates)} Kroner Detekteret')
    axes[1, 2].axis('off')
    
    # Tilføj detaljetekst ved hvert krone-rektangel
    for i, candidate in enumerate(crown_candidates):
        x, y, w, h = candidate['rect']
        confidence = candidate['confidence']
        
        # Beregn position (midten af rektanglet)
        text_x = x + w/2
        text_y = y + h/2
        
        # Farv tekst-baggrund baseret på konfidens
        if confidence >= 0.7:
            facecolor = 'green'
        elif confidence >= 0.5:
            facecolor = 'yellow'
        else:
            facecolor = 'red'
            
        axes[1, 2].text(text_x, text_y, f"{i+1}", color='white', 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(facecolor=facecolor, alpha=0.7, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    plt.suptitle(f'Kronedetektion - {terrain_type}', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    return fig

def process_tile_images_by_terrain():
    """
    Behandler tiles sorteret efter terræntype, med fokus på alle tiles med kroner 
    og et begrænset antal (10) tiles uden kroner for hver terræntype.
    
    Returns:
        dict: Statistikker og resultater
    """
    # Indlæs tileinformation
    tile_info = load_tile_labels()
    if not tile_info:
        print("Ingen tile information indlæst, afslutter")
        return {}
    
    # Opret output-mappe hvis den ikke findes
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Opret dictionaries til at gemme tiles efter terræntype og krone-status
    tiles_with_crowns = defaultdict(list)
    tiles_without_crowns = defaultdict(list)
    
    # Organiser tiles efter terræntype og krone-status
    for filename, info in tile_info.items():
        terrain = info["terrain"]
        crowns = info["crowns"]
        
        # Spring specielle terræntyper over
        if terrain in ["Unknown", "Home", "Table"]:
            continue
        
        # Sortér efter om de har kroner eller ej
        if crowns > 0:
            tiles_with_crowns[terrain].append((filename, info))
        else:
            tiles_without_crowns[terrain].append((filename, info))
    
    # Detektionsresultater
    results = {}
    
    # Statistikvariable for samlet evaluering
    terrain_stats = {}
    
    # Behandl hver terræntype
    for terrain_type in sorted(set(list(tiles_with_crowns.keys()) + list(tiles_without_crowns.keys()))):
        # Hent tiles med kroner for denne terræntype
        has_crown_tiles = tiles_with_crowns.get(terrain_type, [])
        
        # Hent op til 10 tilfældige tiles uden kroner for denne terræntype
        no_crown_tiles = tiles_without_crowns.get(terrain_type, [])
        if len(no_crown_tiles) > 10:
            # Brug et fast seed for reproducerbarhed
            random.seed(42)
            no_crown_tiles = random.sample(no_crown_tiles, 10)
        
        # Kombiner tiles til behandling (alle med kroner + op til 10 uden)
        tiles_to_process = has_crown_tiles + no_crown_tiles
        
        print(f"\nBehandler {len(tiles_to_process)} tiles af terræntypen: {terrain_type}")
        print(f"  - {len(has_crown_tiles)} tiles med kroner")
        print(f"  - {len(no_crown_tiles)} tiles uden kroner")
        
        # Opret output-mappe for denne terræntype
        terrain_output_path = os.path.join(OUTPUT_PATH, terrain_type)
        if not os.path.exists(terrain_output_path):
            os.makedirs(terrain_output_path)
        
        # Tæller for statistik
        correct_detections = 0
        total_detections = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Statistik for tiles med/uden kroner
        has_crown_correct = 0
        has_crown_total = 0
        no_crown_correct = 0
        no_crown_total = 0
        
        # Behandl udvalgte tiles
        for i, (filename, info) in enumerate(tiles_to_process):
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
            detected_crown_count, crown_rects, visualization, confidences = detect_crowns(image, terrain_type)
            
            # Gem resultatet
            results[filename] = {
                "terrain": terrain_type,
                "true_count": true_crown_count,
                "detected_count": detected_crown_count,
                "crown_rects": crown_rects,
                "confidences": confidences
            }
            
            # Opdater statistik
            total_detections += 1
            
            if true_crown_count > 0:
                has_crown_total += 1
                if detected_crown_count > 0:
                    has_crown_correct += 1
            else:
                no_crown_total += 1
                if detected_crown_count == 0:
                    no_crown_correct += 1
            
            if detected_crown_count == true_crown_count:
                correct_detections += 1
                result_text = "✓ KORREKT"
            else:
                result_text = "✗ FEJL"
            
            # Opdater true positives, false positives og false negatives
            if detected_crown_count <= true_crown_count:
                true_positives += detected_crown_count
                false_negatives += (true_crown_count - detected_crown_count)
            else:
                true_positives += true_crown_count
                false_positives += (detected_crown_count - true_crown_count)
            
            print(f"    - Detekteret {detected_crown_count} kroner - {result_text}")
            
            # Gem visualiseringen
            output_filename = f"{os.path.splitext(filename)[0]}_detection.png"
            output_path = os.path.join(terrain_output_path, output_filename)
            visualization.savefig(output_path)
            plt.close(visualization)
        
        # Beregn terrænspecifik statistik
        if total_detections > 0:
            accuracy = correct_detections / total_detections
            
            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            has_crown_accuracy = has_crown_correct / has_crown_total if has_crown_total > 0 else 0
            no_crown_accuracy = no_crown_correct / no_crown_total if no_crown_total > 0 else 0
            
            terrain_stats[terrain_type] = {
                'correct': correct_detections,
                'total': total_detections,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'has_crown_correct': has_crown_correct,
                'has_crown_total': has_crown_total, 
                'has_crown_accuracy': has_crown_accuracy,
                'no_crown_correct': no_crown_correct,
                'no_crown_total': no_crown_total,
                'no_crown_accuracy': no_crown_accuracy
            }
            
            print(f"\n  Statistik for {terrain_type}:")
            print(f"    - Nøjagtig kroneantal: {correct_detections}/{total_detections} ({accuracy:.2%})")
            print(f"    - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
            print(f"    - Detekterer tiles med kroner: {has_crown_correct}/{has_crown_total} ({has_crown_accuracy:.2%})")
            print(f"    - Detekterer tiles uden kroner: {no_crown_correct}/{no_crown_total} ({no_crown_accuracy:.2%})")
    
    # Beregn samlet statistik
    all_correct = sum(stats['correct'] for stats in terrain_stats.values())
    all_total = sum(stats['total'] for stats in terrain_stats.values())
    all_true_positives = sum(stats['true_positives'] for stats in terrain_stats.values())
    all_false_positives = sum(stats['false_positives'] for stats in terrain_stats.values())
    all_false_negatives = sum(stats['false_negatives'] for stats in terrain_stats.values())
    
    all_has_crown_correct = sum(stats['has_crown_correct'] for stats in terrain_stats.values())
    all_has_crown_total = sum(stats['has_crown_total'] for stats in terrain_stats.values())
    all_no_crown_correct = sum(stats['no_crown_correct'] for stats in terrain_stats.values())
    all_no_crown_total = sum(stats['no_crown_total'] for stats in terrain_stats.values())
    
    if all_total > 0:
        overall_accuracy = all_correct / all_total
        overall_precision = all_true_positives / (all_true_positives + all_false_positives) if all_true_positives + all_false_positives > 0 else 0
        overall_recall = all_true_positives / (all_true_positives + all_false_negatives) if all_true_positives + all_false_negatives > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
        
        overall_has_crown_accuracy = all_has_crown_correct / all_has_crown_total if all_has_crown_total > 0 else 0
        overall_no_crown_accuracy = all_no_crown_correct / all_no_crown_total if all_no_crown_total > 0 else 0
        
        print("\n========== SAMLET STATISTIK ==========")
        print(f"Nøjagtig kroneantal: {all_correct}/{all_total} ({overall_accuracy:.2%})")
        print(f"Precision: {overall_precision:.2f}, Recall: {overall_recall:.2f}, F1-score: {overall_f1:.2f}")
        print(f"Detekterer tiles med kroner: {all_has_crown_correct}/{all_has_crown_total} ({overall_has_crown_accuracy:.2%})")
        print(f"Detekterer tiles uden kroner: {all_no_crown_correct}/{all_no_crown_total} ({overall_no_crown_accuracy:.2%})")
        print()
        print("Detaljeret statistik pr. terræntype:")
        for terrain, stats in sorted(terrain_stats.items()):
            print(f"  {terrain}: Nøjagtighed: {stats['accuracy']:.2%}, F1: {stats['f1']:.2f}, Detekterer tiles med kroner: {stats['has_crown_accuracy']:.2%}")
    
    # Tilføj statistik til resultater
    results['_statistics'] = {
        'terrain_stats': terrain_stats,
        'overall': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'has_crown_accuracy': overall_has_crown_accuracy,
            'no_crown_accuracy': overall_no_crown_accuracy
        }
    }
    
    return results

def main():
    """Hovedfunktion til at køre kronedetektion"""
    print("=== Kingdomino Kronedetektion med Forbedret Algoritme ===")
    print("Tester alle tiles med kroner plus 10 tilfældige tiles uden kroner for hver terræntype")
    
    # Behandl tiles efter terræntype
    results = process_tile_images_by_terrain()
    
    print("\nKronedetektion afsluttet. Detaljerede resultater er gemt i:", OUTPUT_PATH)

if __name__ == "__main__":
    main()