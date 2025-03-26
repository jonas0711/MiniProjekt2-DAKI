import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Konstanter
TILE_LABELS_PATH = "Excel+JSON/tile_labels_mapping.json"
EXTRACTED_TILES_PATH = "KingDominoDataset/KingDominoDataset/Extracted_Tiles"
OUTPUT_PATH = "CalibrationResults"

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

def find_crown_samples(tile_info):
    """Finder og grupperer billedfiler med kroner efter terræntype"""
    if not os.path.exists(EXTRACTED_TILES_PATH):
        print(f"Fejl: Extracted tiles-mappen {EXTRACTED_TILES_PATH} blev ikke fundet")
        return {}
    
    crown_samples_by_terrain = {}
    
    # Tjek hver fil i tile_info
    for filename, info in tile_info.items():
        terrain = info["terrain"]
        crowns = info["crowns"]
        
        # Spring specielle terræntyper og tiles uden kroner over
        if terrain in ["Unknown", "Home", "Table"] or crowns == 0:
            continue
        
        # Tjek om filen eksisterer
        image_path = os.path.join(EXTRACTED_TILES_PATH, filename)
        if not os.path.exists(image_path):
            continue
        
        # Tilføj til samples
        if terrain not in crown_samples_by_terrain:
            crown_samples_by_terrain[terrain] = []
        
        crown_samples_by_terrain[terrain].append((image_path, crowns))
    
    # Udskriv resultater
    print("\nFundne eksempler med kroner:")
    for terrain, samples in crown_samples_by_terrain.items():
        print(f"  {terrain}: Fundet {len(samples)} eksempler med kroner")
    
    return crown_samples_by_terrain

def display_hsv_ranges(image, title="HSV Ranges"):
    """Viser forskellige HSV-intervaller på et billede for at hjælpe med kalibrering"""
    # Konverter til HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Opret en række HSV intervaller at teste
    hsv_ranges = [
        ((15, 50, 150), (35, 255, 255), "Standard Guld"),
        ((10, 50, 150), (40, 255, 255), "Bredere Guld"),
        ((20, 80, 180), (30, 255, 255), "Restriktiv Guld"),
        ((15, 50, 100), (35, 255, 255), "Lavere Mætning")
    ]
    
    # Opret en figur med subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Vis originalt billede
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Vis HSV repræsentation
    hsv_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    axes[1].imshow(hsv_rgb)
    axes[1].set_title("HSV Repræsentation")
    axes[1].axis('off')
    
    # Vis forskellige masker
    for i, (lower, upper, label) in enumerate(hsv_ranges[:4]):  # Vis op til 4 ranges
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        
        mask = cv2.inRange(hsv_image, lower_np, upper_np)
        
        # Rens masken
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Vis masken
        axes[i+2].imshow(mask, cmap='gray')
        axes[i+2].set_title(f"{label}\n{lower}-{upper}")
        axes[i+2].axis('off')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    return fig

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

def display_full_calibration(image, crown_count, title="Full Calibration"):
    """Viser en fuld kalibrerings-visualisering inklusiv yderlagsmaske"""
    # Konverter til HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Opret yderlagsmaske
    outer_mask = create_outer_layer_mask(image)
    
    # Opret guldmasker med forskellige parametre
    hsv_ranges = [
        ((15, 50, 150), (35, 255, 255), "Standard Guld"),
        ((10, 50, 150), (40, 255, 255), "Bredere Guld"),
        ((20, 80, 180), (30, 255, 255), "Restriktiv Guld")
    ]
    
    # Opret en figur med subplots - rettet layout til 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Første række - Original, HSV, Yderlagsmaske
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Original ({crown_count} kroner)")
    axes[0, 0].axis('off')
    
    hsv_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    axes[0, 1].imshow(hsv_rgb)
    axes[0, 1].set_title("HSV Repræsentation")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(outer_mask, cmap='gray')
    axes[0, 2].set_title("Yderlag Maske")
    axes[0, 2].axis('off')
    
    # Anden række - Tre forskellige guldmasker
    for i, (lower, upper, label) in enumerate(hsv_ranges):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        
        # Opret guldmaske
        gold_mask = cv2.inRange(hsv_image, lower_np, upper_np)
        
        # Kombiner med yderlagsmaske
        combined_mask = cv2.bitwise_and(gold_mask, outer_mask)
        
        # Rens masken
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Vis kombineret og renset maske
        axes[1, i].imshow(cleaned_mask, cmap='gray')
        axes[1, i].set_title(f"{label}\n{lower}-{upper}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    return fig

def main():
    """Hovedfunktion for HSV kalibrering"""
    print("=== HSV Kalibrering for Kronedetektion ===")
    
    # Opret output-mappe hvis den ikke findes
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Indlæs kroneinformation
    tile_info = load_tile_labels()
    
    if not tile_info:
        print("Ingen tile information indlæst, afslutter")
        return
    
    # Find samples med kroner
    crown_samples_by_terrain = find_crown_samples(tile_info)
    
    if not crown_samples_by_terrain:
        print("Ingen samples med kroner fundet, afslutter")
        return
    
    # Behandl hver terræntype
    for terrain_type, crown_samples in crown_samples_by_terrain.items():
        if not crown_samples:
            continue
        
        print(f"\nKalibrerer for terræntype: {terrain_type}")
        terrain_output_path = os.path.join(OUTPUT_PATH, terrain_type)
        if not os.path.exists(terrain_output_path):
            os.makedirs(terrain_output_path)
        
        # Behandl op til 5 eksempler for hver terræntype
        for i, (image_path, crown_count) in enumerate(crown_samples[:5]):
            # Indlæs billedet
            image = cv2.imread(image_path)
            if image is None:
                print(f"Fejl: Kunne ikke indlæse {image_path}")
                continue
            
            # Konverter fra BGR til RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            filename = os.path.basename(image_path)
            print(f"Kalibrerer på {filename} (med {crown_count} kroner)")
            
            # Vis HSV intervaller
            basic_fig = display_hsv_ranges(image, f"{terrain_type} - {filename} - Grundlæggende Intervaller")
            
            # Gem visualiseringen
            basic_output_path = os.path.join(terrain_output_path, f"{os.path.splitext(filename)[0]}_basic_calibration.png")
            basic_fig.savefig(basic_output_path)
            plt.close(basic_fig)
            
            # Vis fuld kalibrering
            full_fig = display_full_calibration(image, crown_count, f"{terrain_type} - {filename} - Fuld Kalibrering")
            
            # Gem visualiseringen
            full_output_path = os.path.join(terrain_output_path, f"{os.path.splitext(filename)[0]}_full_calibration.png")
            full_fig.savefig(full_output_path)
            plt.close(full_fig)
            
            print(f"  - Gemt kalibreringsvisualiseringer til {terrain_output_path}")

if __name__ == "__main__":
    main()