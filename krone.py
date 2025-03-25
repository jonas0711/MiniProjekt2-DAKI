import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Konstanter
JSON_FILE = "Excel+JSON/tile_labels_mapping.json"
TERRAIN_DIR = "KingDominoDataset/TerrainCategories"
OUTPUT_DIR = "CrownAnalysis"

def load_crown_tiles_data():
    """
    Indlæser data om tiles med kroner fra JSON-filen.
    
    Returns:
        Dict med terræntype → liste af (filnavn, kroneantal) tupler
    """
    # Tjek om JSON-filen eksisterer
    if not os.path.exists(JSON_FILE):
        raise FileNotFoundError(f"JSON-filen {JSON_FILE} findes ikke")
    
    # Indlæs JSON-data
    with open(JSON_FILE, 'r') as f:
        labels_data = json.load(f)
    
    # Opret dictionary til at gemme data grupperet efter terræntype
    terrain_crown_files = defaultdict(list)
    
    # Gennemgå alle brætter
    for board_id, tiles in labels_data.items():
        # Gennemgå alle tiles på dette bræt
        for tile_pos, tile_info in tiles.items():
            terrain = tile_info['terrain']
            crowns = tile_info['crowns']
            filename = tile_info['filename']
            
            # Hvis denne tile har kroner, gem information
            if crowns > 0 and terrain not in ['Home', 'Unknown']:
                terrain_crown_files[terrain].append((filename, crowns))
    
    return terrain_crown_files

def analyze_crown_regions(terrain_crown_files):
    """
    Analyserer farve- og formkarakteristika af kroner for hver terræntype.
    """
    # Opret output-mappe hvis den ikke eksisterer
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Opret mapper til at gemme udklippede kronebilleder
    crown_templates_dir = os.path.join(OUTPUT_DIR, "CrownTemplates")
    if not os.path.exists(crown_templates_dir):
        os.makedirs(crown_templates_dir)
    
    # Dictionary til at gemme HSV-farvespænd for hver terræntype
    hsv_ranges = {}
    
    # Gennemgå hver terræntype
    for terrain, files_info in terrain_crown_files.items():
        print(f"Analyserer {len(files_info)} {terrain}-tiles med kroner...")
        
        # Opret mappe for denne terræntype
        terrain_dir = os.path.join(crown_templates_dir, terrain)
        if not os.path.exists(terrain_dir):
            os.makedirs(terrain_dir)
        
        # Lister til at gemme HSV-værdier
        h_values = []
        s_values = []
        v_values = []
        
        # Lister til at gemme udklippede kroner
        crown_templates = []
        crown_counter = 0
        
        # Gennemgå hver fil for denne terræntype
        for idx, (filename, crown_count) in enumerate(files_info):
            # Find den fulde sti til billedet
            tile_path = os.path.join("KingDominoDataset/KingDominoDataset/Extracted_Tiles", filename)
            if not os.path.exists(tile_path):
                print(f"Advarsel: Filen {tile_path} blev ikke fundet. Springer over.")
                continue
            
            # Indlæs billedet
            image = cv2.imread(tile_path)
            if image is None:
                print(f"Advarsel: Kunne ikke indlæse billedet {tile_path}. Springer over.")
                continue
            
            # Konverter fra BGR til RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Opdel billedet i 5x5 regioner
            height, width, _ = image_rgb.shape
            region_h = height // 5
            region_w = width // 5
            
            # Konverter billedet til HSV for farveanalyse
            image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # Fokuser på de yderste regioner (hvor kroner typisk er)
            outer_regions = []
            for i in range(5):
                for j in range(5):
                    # Spring den midterste 3x3 region over (1:4, 1:4)
                    if 1 <= i <= 3 and 1 <= j <= 3:
                        continue
                    
                    # Beregn regionens koordinater
                    y_start = i * region_h
                    y_end = (i + 1) * region_h
                    x_start = j * region_w
                    x_end = (j + 1) * region_w
                    
                    # Udskær regionen
                    region = image_rgb[y_start:y_end, x_start:x_end]
                    region_hsv = image_hsv[y_start:y_end, x_start:x_end]
                    
                    outer_regions.append((region, region_hsv, (x_start, y_start, x_end, y_end)))
            
            # Anvend en guldfarvemaske for at identificere potentielle kroner
            # Startværdier baseret på den eksisterende implementation
            lower_gold = np.array([15, 100, 100])  # Bredere interval til start
            upper_gold = np.array([35, 255, 255])
            
            for region, region_hsv, coords in outer_regions:
                x_start, y_start, x_end, y_end = coords
                
                # Lav en maske for guldfarver
                mask = cv2.inRange(region_hsv, lower_gold, upper_gold)
                
                # Anvend morfologiske operationer for at forbedre masken
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                # Find konturer
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filtrer konturer baseret på areal og form
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 50 or area > 800:  # Justerbare tærskelværdier
                        continue
                    
                    # Beregn cirkularitet
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.3:  # Justerbar tærskelværdi
                        continue
                    
                    # Vi har fundet en potentiel krone
                    # Lav en maske kun for denne kontur
                    crown_mask = np.zeros_like(mask)
                    cv2.drawContours(crown_mask, [contour], -1, 255, -1)
                    
                    # Beregn bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Udskær krone-området med lidt padding
                    padding = 5
                    x_min = max(0, x - padding)
                    y_min = max(0, y - padding)
                    x_max = min(region.shape[1], x + w + padding)
                    y_max = min(region.shape[0], y + h + padding)
                    
                    # Konverter til globale koordinater i det oprindelige billede
                    global_x_min = x_start + x_min
                    global_y_min = y_start + y_min
                    global_x_max = x_start + x_max
                    global_y_max = y_start + y_max
                    
                    # Udskær kronetemplate fra originalbilledet
                    crown_template = image_rgb[global_y_min:global_y_max, global_x_min:global_x_max]
                    
                    # Gem kun hvis det er et rimeligt stort område
                    if crown_template.shape[0] > 10 and crown_template.shape[1] > 10:
                        # Gem som template
                        template_filename = f"{terrain}_crown_{crown_counter}.png"
                        template_path = os.path.join(terrain_dir, template_filename)
                        cv2.imwrite(template_path, cv2.cvtColor(crown_template, cv2.COLOR_RGB2BGR))
                        crown_counter += 1
                        
                        # Udtræk HSV-værdier
                        hsv_values = region_hsv[y:y+h, x:x+w][crown_mask[y:y+h, x:x+w] == 255]
                        if len(hsv_values) > 0:
                            h_values.extend(hsv_values[:, 0])
                            s_values.extend(hsv_values[:, 1])
                            v_values.extend(hsv_values[:, 2])
        
        # Beregn HSV-statistik
        if len(h_values) > 0:
            h_min, h_max = np.percentile(h_values, [5, 95])
            s_min, s_max = np.percentile(s_values, [5, 95])
            v_min, v_max = np.percentile(v_values, [5, 95])
            
            hsv_ranges[terrain] = {
                'lower': np.array([int(h_min), int(s_min), int(v_min)]),
                'upper': np.array([int(h_max), int(s_max), int(v_max)])
            }
            
            print(f"  Fundne kroner: {crown_counter}")
            print(f"  HSV-interval for {terrain}: H=[{h_min:.1f}, {h_max:.1f}], S=[{s_min:.1f}, {s_max:.1f}], V=[{v_min:.1f}, {v_max:.1f}]")
            
            # Visualiser HSV-fordelinger
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.hist(h_values, bins=30, color='r', alpha=0.7)
            plt.title(f"{terrain} - Hue Distribution")
            plt.xlabel("Hue Value")
            plt.ylabel("Frequency")
            
            plt.subplot(132)
            plt.hist(s_values, bins=30, color='g', alpha=0.7)
            plt.title(f"{terrain} - Saturation Distribution")
            plt.xlabel("Saturation Value")
            
            plt.subplot(133)
            plt.hist(v_values, bins=30, color='b', alpha=0.7)
            plt.title(f"{terrain} - Value Distribution")
            plt.xlabel("Value")
            
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"{terrain}_hsv_distributions.png")
            plt.savefig(plot_path)
            plt.close()
    
    # Gem HSV-intervaller til en JSON-fil
    hsv_ranges_json = {}
    for terrain, ranges in hsv_ranges.items():
        hsv_ranges_json[terrain] = {
            'lower': ranges['lower'].tolist(),
            'upper': ranges['upper'].tolist()
        }
    
    with open(os.path.join(OUTPUT_DIR, "crown_hsv_ranges.json"), 'w') as f:
        json.dump(hsv_ranges_json, f, indent=4)
    
    return hsv_ranges

def create_crown_template_tester(hsv_ranges):
    """
    Opretter et simpelt test-script til at afprøve de genererede HSV-intervaller.
    """
    test_script_path = os.path.join(OUTPUT_DIR, "test_crown_detection.py")
    
    script_content = """import cv2
import numpy as np
import json
import os
import argparse

def test_crown_detection(image_path, terrain_type=None):
    # Indlæs HSV-intervaller
    with open("CrownAnalysis/crown_hsv_ranges.json", 'r') as f:
        hsv_ranges = json.load(f)
    
    # Indlæs billedet
    image = cv2.imread(image_path)
    if image is None:
        print(f"Kunne ikke indlæse billedet {image_path}")
        return
    
    # Konverter til HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Opret en output-mappe til resultater
    output_dir = "CrownAnalysis/TestResults"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Opret output-billede til at vise alle detektioner
    result_image = image.copy()
    
    # Hvis terræntype er angivet, test kun den type
    if terrain_type and terrain_type in hsv_ranges:
        terrain_types = [terrain_type]
    else:
        terrain_types = hsv_ranges.keys()
    
    for terrain in terrain_types:
        print(f"Tester HSV-interval for {terrain}...")
        
        # Få HSV-intervallet
        lower = np.array(hsv_ranges[terrain]['lower'])
        upper = np.array(hsv_ranges[terrain]['upper'])
        
        # Opret maske
        mask = cv2.inRange(hsv, lower, upper)
        
        # Anvend morfologiske operationer
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find konturer
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer konturer baseret på areal og form
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 800:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3:
                continue
            
            # Tegn kontur på resultatet
            color = (0, 255, 0) if terrain == "Forest" else \\
                    (0, 0, 255) if terrain == "Field" else \\
                    (255, 0, 0) if terrain == "Lake" else \\
                    (255, 255, 0)
            
            cv2.drawContours(result_image, [contour], -1, color, 2)
            
            # Få centrum
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Tegn label
                cv2.putText(result_image, terrain, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Gem resultatet
    base_filename = os.path.basename(image_path)
    result_path = os.path.join(output_dir, f"detected_{base_filename}")
    cv2.imwrite(result_path, result_image)
    
    print(f"Resultat gemt til {result_path}")
    return result_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test crown detection')
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--terrain', help='Specific terrain type to test')
    
    args = parser.parse_args()
    test_crown_detection(args.image, args.terrain)
"""
    
    with open(test_script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Test-script oprettet: {test_script_path}")

def main():
    """
    Hovedfunktion til at køre hele analysen.
    """
    print("Starter analyse af kroner i Kingdomino-tiles...")
    
    # Indlæs data om tiles med kroner
    terrain_crown_files = load_crown_tiles_data()
    
    # Vis statistik over data
    print("\nStatistik over kroner pr. terræntype:")
    for terrain, files in terrain_crown_files.items():
        total_crowns = sum(crown_count for _, crown_count in files)
        print(f"  {terrain}: {len(files)} tiles med i alt {total_crowns} kroner")
    
    # Analyser kroneregioner og farver
    hsv_ranges = analyze_crown_regions(terrain_crown_files)
    
    # Opret test-script
    create_crown_template_tester(hsv_ranges)
    
    print("\nAnalyse fuldført.")
    print(f"Resultater gemt i {OUTPUT_DIR}-mappen.")
    print("Du kan nu bruge genererede HSV-værdier til at forbedre kronedetektion.")

if __name__ == "__main__":
    main()