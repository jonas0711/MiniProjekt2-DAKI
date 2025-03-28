import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from model import load_model, extract_features
import re

# Konstanter
MODEL_PATH = "kingdomino_terrain_model.pkl"
TERRAIN_FEATURES_PATH = "Visualiseringer/terrain_color_features.json"
TILE_LABELS_PATH = "Excel+JSON/tile_labels_mapping.json"
CATEGORIES_PATH = "KingDominoDataset/TerrainCategories"
OUTPUT_PATH = "CrownDetectionResults"

# HSV intervaller for guld/kroner - skal justeres for hvert terræn
DEFAULT_GOLD_HSV_RANGE = {
    'Field': [(20, 60, 180), (40, 255, 255)],     # Mere restriktivt pga. gullig baggrund
    'Forest': [(15, 80, 150), (35, 255, 255)],    # Standardinterval
    'Lake': [(15, 80, 150), (40, 255, 255)],      # Standardinterval
    'Mine': [(15, 50, 150), (45, 255, 255)],      # Bredere interval pga. mørk baggrund
    'Swamp': [(15, 80, 150), (35, 255, 255)],     # Standardinterval
    'Grassland': [(15, 80, 150), (35, 255, 255)]  # Standardinterval
}

def load_terrain_features():
    """Indlæser terrænfarve-features fra json-fil"""
    if not os.path.exists(TERRAIN_FEATURES_PATH):
        print(f"Advarsel: Terrænfeatures-fil {TERRAIN_FEATURES_PATH} ikke fundet")
        return {}
    
    with open(TERRAIN_FEATURES_PATH, 'r') as f:
        terrain_features = json.load(f)
    
    print(f"Indlæst terrænfarve-features for {len(terrain_features)} terræntyper")
    return terrain_features

def load_crown_ground_truth():
    """Indlæser ground truth kroneantal fra json-fil"""
    if not os.path.exists(TILE_LABELS_PATH):
        print(f"Advarsel: Tile-labels-fil {TILE_LABELS_PATH} ikke fundet")
        return {}
    
    with open(TILE_LABELS_PATH, 'r') as f:
        tile_labels = json.load(f)
    
    # Udtræk kroneantal for hver tile
    crown_counts = {}
    for board_name, tiles in tile_labels.items():
        for tile_pos, tile_info in tiles.items():
            filename = tile_info["filename"]
            crowns = tile_info["crowns"]
            crown_counts[filename] = crowns
    
    print(f"Indlæst ground truth kroneantal for {len(crown_counts)} tiles")
    return crown_counts

def extract_board_and_position(filename):
    """Udtrækker bræt-id og position fra filnavn"""
    match = re.search(r'^(\d+)_tile_(\d+)_(\d+)', filename)
    if match:
        board_id = match.group(1)
        row = match.group(2)
        col = match.group(3)
        return board_id, row, col
    return None, None, None

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

def detect_crowns_in_tile(image, terrain_type, terrain_features):
    """Detekterer kroner i en tile baseret på terræntype"""
    # Konverter til HSV (bedre for farvedetektion)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Hent terrænspecifikke farveintervaller for guld
    gold_hsv_range = DEFAULT_GOLD_HSV_RANGE.get(terrain_type, [(15, 80, 150), (35, 255, 255)])
    lower_gold = np.array(gold_hsv_range[0])
    upper_gold = np.array(gold_hsv_range[1])
    
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
    
    # Find sammenhængende komponenter (BLOBs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)
    
    # Vi starter ved 1 for at skippe baggrunden (label 0)
    crown_candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Grundlæggende filtreringskriterier baseret på størrelse
        min_crown_size = 30  # Justér baseret på billedstørrelse
        max_crown_size = 500  # Justér baseret på billedstørrelse
        
        if min_crown_size < area < max_crown_size:
            # Beregn aspect ratio (bredde/højde)
            aspect_ratio = w / h if h > 0 else 0
            
            # Kroner er normalt nogenlunde cirkulære eller kvadratiske
            if 0.5 < aspect_ratio < 2.0:
                # Ekstrahér blob-regionen for at beregne flere features
                blob_region = np.zeros_like(labels, dtype=np.uint8)
                blob_region[labels == i] = 255
                
                # Beregn kantdensitet (anvender Sobel operatoren)
                gray_region = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # Anvend masken
                masked_gray = cv2.bitwise_and(gray_region, gray_region, mask=blob_region)
                
                # Beregn gradient
                gradient_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
                
                # Beregn gennemsnitlig kantintensitet
                non_zero_pixels = np.count_nonzero(blob_region)
                if non_zero_pixels > 0:
                    edge_density = np.sum(gradient_magnitude) / non_zero_pixels
                    
                    # Acceptér blob som krone hvis kantdensiteten er over en tærskel
                    edge_threshold = 10  # Justér baseret på tests
                    if edge_density > edge_threshold:
                        crown_candidates.append((x, y, w, h, area, edge_density))
    
    # Hvis der er flere kandidater end forventet, behold kun de stærkeste
    # baseret på kantdensitet (som er en god indikator for kronepotentiale)
    if len(crown_candidates) > 0:
        # Sortér efter kantdensitet (højeste først)
        crown_candidates.sort(key=lambda x: x[5], reverse=True)
        
    # Filtrér overlappende kandidater (simple non-max suppression)
    filtered_candidates = []
    for candidate in crown_candidates:
        x1, y1, w1, h1 = candidate[:4]
        
        # Tjek om denne kandidat overlapper væsentligt med allerede accepterede kandidater
        overlapping = False
        for accepted in filtered_candidates:
            x2, y2, w2, h2 = accepted[:4]
            
            # Beregn overlap
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            # Hvis overlap er væsentligt, afvis kandidaten
            min_area = min(w1 * h1, w2 * h2)
            if overlap_area > 0.5 * min_area:
                overlapping = True
                break
        
        if not overlapping:
            filtered_candidates.append(candidate)
    
    # Antal kroner er lig med antal accepterede kandidater
    crown_count = len(filtered_candidates)
    
    # Her kunne vi justere antallet baseret på terræntype (nogle terræntyper 
    # har typisk færre kroner end andre)
    
    # Lav visualisering af resultatet
    visualization = create_detection_visualization(image, hsv_image, gold_mask, 
                                                   outer_layer_mask, combined_mask, 
                                                   cleaned_mask, filtered_candidates)
    
    return crown_count, filtered_candidates, visualization

def create_detection_visualization(image, hsv_image, gold_mask, outer_mask, 
                                   combined_mask, cleaned_mask, crown_candidates):
    """Skaber en visuel fremstilling af detektionsprocessen"""
    # Konverter HSV tilbage til RGB for visualisering
    hsv_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    # Opret figur med subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original billede
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Tile')
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
    
    # Kombineret maske
    axes[1, 1].imshow(cleaned_mask, cmap='gray')
    axes[1, 1].set_title('Renset Maske')
    axes[1, 1].axis('off')
    
    # Resultat med markerede kroner
    result_img = image.copy()
    for x, y, w, h, area, _ in crown_candidates:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    axes[1, 2].imshow(result_img)
    axes[1, 2].set_title(f'Detekterede Kroner: {len(crown_candidates)}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig

def evaluate_crown_detection(results, ground_truth):
    """Evaluerer nøjagtigheden af kronedetektion i forhold til ground truth"""
    correct = 0
    total = 0
    
    for filename, detected_count in results.items():
        if filename in ground_truth:
            true_count = ground_truth[filename]
            total += 1
            if detected_count == true_count:
                correct += 1
                print(f"{filename}: KORREKT - Detekteret: {detected_count}, Ground Truth: {true_count}")
            else:
                print(f"{filename}: FEJL - Detekteret: {detected_count}, Ground Truth: {true_count}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nEvalueringsresultat: {correct}/{total} korrekte detektioner ({accuracy:.2%} nøjagtighed)")
    return accuracy

def main():
    """Hovedfunktion for kronedetektion"""
    print("=== Kingdomino Kronedetektion ===")
    
    # Opret output-mappe hvis den ikke findes
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"Oprettet output-mappe: {OUTPUT_PATH}")
    
    # Indlæs terrænklassifikationsmodel
    print("Indlæser terrænklassifikationsmodel...")
    model = load_model(MODEL_PATH)
    
    # Indlæs terrænfarve-features
    terrain_features = load_terrain_features()
    
    # Indlæs ground truth kroneantal
    crown_ground_truth = load_crown_ground_truth()
    
    # Håndter hvert terræn separat
    results = {}
    
    for terrain_type in os.listdir(CATEGORIES_PATH):
        # Spring over hvis det ikke er en mappe
        terrain_path = os.path.join(CATEGORIES_PATH, terrain_type)
        if not os.path.isdir(terrain_path):
            continue
        
        print(f"\nBehandler terræntype: {terrain_type}")
        terrain_output_path = os.path.join(OUTPUT_PATH, terrain_type)
        if not os.path.exists(terrain_output_path):
            os.makedirs(terrain_output_path)
        
        # Gennemgå alle billeder i terrænmappen
        for image_file in os.listdir(terrain_path):
            # Kun behandl billedfiler
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(terrain_path, image_file)
            
            # Indlæs billedet
            image = cv2.imread(image_path)
            if image is None:
                print(f"Fejl: Kunne ikke indlæse {image_path}")
                continue
            
            # Konverter fra BGR til RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print(f"Behandler billede: {image_file}")
            
            # Detektér kroner i tilen
            crown_count, candidates, visualization = detect_crowns_in_tile(
                image, terrain_type, terrain_features
            )
            
            # Gem resultatet
            results[image_file] = crown_count
            
            # Gem visualiseringen
            visualization_path = os.path.join(terrain_output_path, f"{image_file.split('.')[0]}_detection.png")
            visualization.savefig(visualization_path)
            plt.close(visualization)
            
            # Vis resultat i terminalen
            print(f"  - Detekteret {crown_count} kroner")
            
            # Vis ground truth hvis tilgængelig
            if image_file in crown_ground_truth:
                true_count = crown_ground_truth[image_file]
                print(f"  - Ground truth: {true_count} kroner")
                if crown_count == true_count:
                    print(f"  - ✓ KORREKT")
                else:
                    print(f"  - ✗ FEJL")
                    
    # Evaluer resultaterne
    print("\n=== Evaluering af Kronedetektion ===")
    evaluate_crown_detection(results, crown_ground_truth)

if __name__ == "__main__":
    main()